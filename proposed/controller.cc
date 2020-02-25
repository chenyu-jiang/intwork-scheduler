// Copyright 2019 Uber Technologies, Inc. All Rights Reserved.
// Modifications copyright Microsoft
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#include "controller.h"

#include <atomic>
#include <map>
#include <queue>
#include <set>
#include <thread>
#include <chrono>
#include <unordered_set>

namespace proposed {
namespace common {

// Controller ==================================================================

bool Controller::IncrementTensorCount_(const Request& msg) {
  if(msg.request_type() != Request::TENSOR_READY)
    throw std::runtime_error(
      "Internal: msg with type PARTITION_FINISHED in IncrementTensorCount_");
  ready_table_[msg.tensor_id()] ++;
  if (ready_table_[msg.tensor_id()] == world_size_) {
    ready_table_[msg.tensor_id()] = 0;
    return true;
  } else return false;
}

void Controller::ProcessRequests_(const std::vector<Request>& recvd_requests){
  for(auto& req: recvd_requests) {
    if(req.request_type() == Request::TENSOR_READY) {
      // Received tensor ready message
      bool all_ready = IncrementTensorCount_(req);
      if(all_ready) {
        // Push copies of tensorpack of this tensor into executor
        for(auto tp: tp_table_.at(req.tensor_id())) {
          pack_executor_.Post(tp);
        }
      }
    } else {
      // Received partition finished message
      pack_executor_.SignalPushFinished();
      if(tp_count_table_[req.tensor_id()] == 0) {
        // entire tensor execution finished
        DeleteFromTPTable(req.tensor_id());
        tensor_manager_.DeleteTensorWithID(req.tensor_id());
      }
    }
  }
}

void
Controller::ProcessResponses_(const std::vector<Response>& recvd_responses) {
  for(auto& res: recvd_responses) {
    if (res.response_type() == Response::ERROR) {
      throw std::runtime_error("Response of type ERROR received.");
    }
    // response of type release, call the tensor manager
    tensor_manager_.ReleaseTensor(res.tensor_id(), res.partition_id());
  }
}

void Controller::ConstructTensorPacks_(int32_t tensor_id) {
  int32_t num_partitions = tensor_manager_.GetNumPartitions(tensor_id);
  // sanity check
  assert(num_partitions % world_size_ == 0);

  int32_t priority = tensor_manager_.GetPriority(tensor_id);
  int32_t num_packs = num_partitions / world_size_;

  for(int32_t pack_id=0; pack_id < num_packs; pack_id++) {
    std::vector<std::vector<TensorPackElement>> pack;
    for(int32_t step_id=0; step_id < world_size_; step_id++) {
      std::vector<TensorPackElement> step_elements;
      for(int32_t worker_id=0; worker_id<world_size_; worker_id++) {
        int32_t current_partition_id = 
          pack_id*world_size_+ ((step_id+worker_id) % world_size_);
        step_elements.emplace_back(
          TensorPackElement(worker_id, tensor_id, current_partition_id));
      }
      pack.emplace_back(std::move(step_elements));
    }
    tp_table_[tensor_id].emplace_back(
      TensorPack(pack, priority, 
                [this, tensor_id]{this->tp_count_table_[tensor_id] --;} ));
  }
  tp_count_table_[tensor_id] = num_packs;
}

void Controller::DeleteFromTPTable(int32_t tensor_id) {
  tp_table_.erase(tensor_id);
  tp_count_table_.erase(tensor_id);
}

void Controller::RunLoopOnce_() {
  // Send/Receive requests
  std::vector<Request> reqs;
  if (is_coordinator_) {
    reqs = RecvRequests_();
  } else {
    SendRequests_(tensor_manager_.ResetRequestList());
  }
  // Process requests on coordinator
  if (is_coordinator_) {
    ProcessRequests_(reqs);
    ProcessResponses_(SendResponses_(tensor_manager_.ResetResponseList()));
  } else {
    // Process responses on workers
    ProcessResponses_(RecvResponses_());
  }
}

void Controller::LaunchBackGroundThread() {
  static bool thread_launched = false;
  if (thread_launched == true) {
    throw std::runtime_error("BG Thread cannot be launched twice.");
  }
  bg_thread = std::thread(&Controller::RunMainLoop_, this);
  thread_launched = true;
}

void Controller::RunMainLoop_() {
  while(true) {
    RunLoopOnce_();
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
}

void Controller::PostTensor(std::vector<Tensor>& tensors, int32_t priority) {
  tensor_manager_.PostTensor(tensors, priority);
  if (is_coordinator_) ConstructTensorPacks_(tensors[0].tensor_id);
}

void 
Controller::SignalPartitionFinished(int32_t tensor_id, int32_t partition_id) {
  tensor_manager_.SignalPartitionFinished(tensor_id, partition_id);
}

void Controller::Log(std::string str) {
  std::cout << "[" << rank_ << "] " << str << std::endl;
}

// TensorPack ==================================================================

bool TensorPack::IncrementReceivedCount() {
    if(count_finished_ >= num_workers_) 
      throw std::runtime_error("Received count already reached worker number.");
    count_finished_ ++;
    if(count_finished_ == num_workers_) return true;
    else return false;
}

const std::vector<TensorPackElement> TensorPack::YieldTensorPackElements() {
    if (current_step_ >= num_workers_) {
      return std::vector<TensorPackElement>();
    }
    current_step_ ++;
    count_finished_ = 0;
    return tensors_[current_step_-1];
}

std::string TensorPack::to_string() const {
  std::string out = "";
  for(const auto& element_vec: tensors_) {
    out += "\n";
    for(const auto& t: element_vec) {
      out += t.to_string();
    }
  }
  return out;
}

// PackExecutor ================================================================

Status PackExecutor::SignalPushFinished() {
  if(executing_ == false)
    throw std::runtime_error(
      "PushFinished signal received while executor is not executing.");
  
  bool step_finished = executing_tp_.IncrementReceivedCount();
  if(step_finished) {
    // Check if we need to send response messages to workers
    const std::vector<TensorPackElement> next_tensors = 
      executing_tp_.YieldTensorPackElements();
    if(! next_tensors.empty()) {
      // There's more partitions to send
      std::vector<Response> responses = ConstructResponse(next_tensors);
      for(auto& res: responses) {
        tensor_manager_.AddResponseToList(res, res.rank());
      }
    } else {
      // TP execution complete
      executing_ = false;
      executing_tp_.on_finish();
      ExecutePackInQueue_();
    }
  }
  return Status::OK();
}

bool PackExecutor::ExecutePackInQueue_() {
  if (executing_) return false;

  if(! tp_queue_.empty()) {
    executing_tp_ = tp_queue_.top();
    executing_ = true;
    tp_queue_.pop();
    std::vector<Response> responses = 
      ConstructResponse(executing_tp_.YieldTensorPackElements());
    for(auto& res: responses) {
      tensor_manager_.AddResponseToList(res, res.rank());
    }
    return true;
  } else return false;
}

std::vector<Response> 
PackExecutor::ConstructResponse(const std::vector<TensorPackElement>& tensors) 
const {
  std::vector<Response> res;
  for(auto& t: tensors) {
    res.emplace_back(Response(Response::RELEASE, t.rank, 
                      t.tensor_id, t.partition_id));
  }
  return res;
}

void PackExecutor::Post(TensorPack pack) {
  tp_queue_.emplace(std::move(pack));
  ExecutePackInQueue_();
}

} // namespace common
} // namespace proposed
