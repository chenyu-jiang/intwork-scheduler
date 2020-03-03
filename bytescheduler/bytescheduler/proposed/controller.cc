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
#include <cassert>

namespace proposed {
namespace common {

// Controller ==================================================================

bool Controller::IncrementTensorCount_(const Request& msg) {
  if(msg.request_type() != Request::TENSOR_READY)
    assert( false && "Internal: msg with type PARTITION_FINISHED in IncrementTensorCount_");
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
      // Log("Received ready message from "+std::to_string(req.tensor_id()) + ".");
      bool all_ready = IncrementTensorCount_(req);
      if(all_ready) {
        if (stp_table_.find(req.tensor_id()) != stp_table_.end()) {
          // is small tensor
          small_tensor_executor_.Post(stp_table_.at(req.tensor_id()));
        } else {
          // Not a small tensor
          // Push copies of tensorpack of this tensor into executor
          for(auto tp: tp_table_.at(req.tensor_id())) {
            pack_executor_.Post(tp);
          }
        }
      }
    } else {
      // Received partition finished message
      if(stp_table_.find(req.tensor_id()) != stp_table_.end()) {
        // is small tensor
        small_tensor_executor_.SignalPushFinished(req.tensor_id());
      } else {
        pack_executor_.SignalPushFinished();
      }
      if(tp_count_table_[req.tensor_id()] == 0) {
        // entire tensor execution finished
        Log("Deleting tensor " + std::to_string(req.tensor_id()) + " from table.");
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
      assert(false && "Response of type ERROR received.");
    }
    // response of type release, call the tensor manager
    // Log("Releasing tensor i: "+ std::to_string(res.tensor_id()) + ",p: " +std::to_string(res.partition_id()));
    tensor_manager_.ReleaseTensor(res.tensor_id(), res.partition_id());
  }
}

void Controller::ConstructTensorPacks_(int32_t tensor_id) {
  int32_t num_partitions = tensor_manager_.GetNumPartitions(tensor_id);
  
  if(num_partitions == 1) {
    // Small tensor
    int32_t priority = tensor_manager_.GetPriority(tensor_id);
    int32_t assigned_server = tensor_manager_.GetAssignedServer(tensor_id);
    if(assigned_server == -1)
      assert(false && "Unregistered assigned server for small tensor in tensor_manager.");
    std::vector<TensorPackElement> spack;
    for(int32_t worker_id = 0; worker_id < world_size_; worker_id++) {
      spack.emplace_back(TensorPackElement(worker_id, tensor_id, 0));
    }
    stp_table_[tensor_id] = 
      SmallTensorPack(spack, priority, [this, tensor_id]{this->tp_count_table_[tensor_id] --;}, assigned_server);
    tp_count_table_[tensor_id] = 1;
  } else {
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
}

void Controller::DeleteFromTPTable(int32_t tensor_id) {
  if(stp_table_.find(tensor_id) != stp_table_.end()) {
    stp_table_.erase(tensor_id);
  } else {
    tp_table_.erase(tensor_id);
  }
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
    assert(false && "BG Thread cannot be launched twice.");
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

void Controller::PostTensor(std::vector<Tensor>& tensors, int32_t priority, int32_t assigned_server) {
  // Log("In Controller: Posted tensor with tensor id "+ std::to_string(tensors[0].tensor_id));
  tensor_manager_.PostTensor(tensors, priority, assigned_server);
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
      assert( false && "Received count already reached worker number.");
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

// SmallTensorPack =============================================================

void SmallTensorPack::SetFirstWorker(int32_t worker_id) {
  if(worker_id >= num_workers_) assert( false && "Worker id does not exist.");
  worker_offset_ = worker_id;
}

const TensorPackElement SmallTensorPack::YieldTensorPackElements() {
    if (current_step_ >= num_workers_) {
      return TensorPackElement();
    }
    current_step_ ++;
    return tensors_[(current_step_ + worker_offset_ - 1) % num_workers_];
}

std::string SmallTensorPack::to_string() const {
  std::string out = "";
  for(const auto& element_vec: tensors_) {
    out += "\n";
    out += element_vec.to_string();
  }
  return out;
}

// PackExecutor ================================================================

Status PackExecutor::SignalPushFinished() {
  if(executing_ == false)
    assert( false && "PushFinished signal received while executor is not executing.");
  
  bool step_finished = executing_tp_.IncrementReceivedCount();
  if(step_finished) {
    // Check if we need to send response messages to workers
    const std::vector<TensorPackElement> next_tensors = 
      executing_tp_.YieldTensorPackElements();
    if(! next_tensors.empty()) {
      // There's more partitions to send
      std::vector<Response> responses = ConstructResponse_(next_tensors);
      for(auto& res: responses) {
        tensor_manager_.AddResponseToList(res);
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
      ConstructResponse_(executing_tp_.YieldTensorPackElements());
    for(auto& res: responses) {
      tensor_manager_.AddResponseToList(res);
    }
    return true;
  } else return false;
}

std::vector<Response> 
PackExecutor::ConstructResponse_(const std::vector<TensorPackElement>& tensors) 
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


// SmallTensorExecutor =========================================================

void SmallTensorExecutor::Finalize() {
  if(! tensor_manager_.IsFinalized()) 
    assert(false && "Must finalize tensor manager before initializing SmallTensorExecutor.");
  world_size_ = tensor_manager_.GetWorldSize();
  executing_tps_.resize(world_size_);

  for(int i=0; i<world_size_;i++) {
    idle_servers.insert(i);
  }
  finalized_ = true;
}

inline void SmallTensorExecutor::CheckFinalized() {
  if(! finalized_)  assert(false && "Must finalize SmallTensorExecutor before use.");
}

Status SmallTensorExecutor::SignalPushFinished(int32_t tensor_id) {
  CheckFinalized();
  if(tensor_id_to_server_dict_.find(tensor_id) == tensor_id_to_server_dict_.end())
    assert( false && "PushFinished signal received while tensor is not executing.");
    
  int32_t assigned_server = tensor_id_to_server_dict_[tensor_id];

  TensorPackElement next_tensor = 
    executing_tps_[assigned_server].YieldTensorPackElements();
  if(next_tensor.rank != -1) {
    // There's more partitions to send
    Response response = ConstructResponse_(next_tensor);
    tensor_manager_.AddResponseToList(response);
  } else {
    // TP execution complete
    idle_servers.insert(assigned_server);
    executing_tps_[assigned_server].on_finish();
    ExecutePackInQueue_();
  }
  return Status::OK();
}

bool SmallTensorExecutor::ExecutePackInQueue_() {
  if (idle_servers.empty()) return false;

  if(! tp_queue_.empty()) {
    bool executed_stp = false;
    std::vector<SmallTensorPack> tmp_q;
    while(!tp_queue_.empty()) {
      const SmallTensorPack& stp = tp_queue_.top();
      int32_t assigned_server = stp.get_assigned_server();
      if(idle_servers.find(assigned_server) == idle_servers.end()) {
        tmp_q.push_back(stp);
        tp_queue_.pop();
      } else {
        // got a tensor that matches an empty position
        executing_tps_[assigned_server] = stp;
        idle_servers.erase(assigned_server);
        tensor_id_to_server_dict_[stp.get_tensor_id()] = assigned_server;
        // assign worker offset
        executing_tps_[assigned_server].SetFirstWorker(assigned_server);

        tp_queue_.pop();
        Response res = 
          ConstructResponse_(
            executing_tps_[assigned_server].YieldTensorPackElements()
            );
        tensor_manager_.AddResponseToList(res);
        executed_stp = true;
        break;
      }
    }
    for(auto& stp: tmp_q) {
      tp_queue_.push(stp);
    }
    if (executed_stp) {
      ExecutePackInQueue_();
      return true;
    } else return false;
  } else return false;
}

Response
SmallTensorExecutor::ConstructResponse_(const TensorPackElement& t) 
const {
  return Response(Response::RELEASE, t.rank, 
                      t.tensor_id, t.partition_id);
}

void SmallTensorExecutor::Post(SmallTensorPack pack) {
  CheckFinalized();
  tp_queue_.emplace(std::move(pack));
  ExecutePackInQueue_();
}

} // namespace common
} // namespace proposed
