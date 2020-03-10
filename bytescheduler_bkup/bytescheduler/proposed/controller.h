// Copyright 2019 Uber Technologies, Inc. All Rights Reserved.
// Modifications Chenyu Jiang
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

#ifndef PROPOSED_CONTROL_MANAGER_H
#define PROPOSED_CONTROL_MANAGER_H

#include <iostream>
#include <queue>
#include <vector>
#include <thread>
#include <mutex>
#include <unordered_set>
#include "tensor_manager.h"
#include "common.h"

namespace proposed {
namespace common {

class TensorPackElement {
public:
  TensorPackElement() {};

  TensorPackElement(int32_t rank, int32_t tensor_id, int32_t partition_id): 
    rank(rank), tensor_id(tensor_id), partition_id(partition_id) {}

  TensorPackElement(const TensorPackElement& other) {
    tensor_id = other.tensor_id;
    rank = other.rank;
    partition_id = other.partition_id;
  }

  std::string to_string() const {
    return "r: " + std::to_string(rank) + 
            "l: " + std::to_string(tensor_id);
            ", p: " + std::to_string(partition_id) + "; ";
  }

  int32_t rank = -1;
  int32_t tensor_id = -1;
  int32_t partition_id = -1;
};

using PackExecutionCallback = std::function<void()>;

class TensorPack {
public:
  struct compare_tp {
    bool operator()(const TensorPack& l, const TensorPack& r) {
      return l.priority > r.priority;
    }
  };

  TensorPack() {};

  TensorPack(const std::vector<std::vector<TensorPackElement>>& tensors, 
            int32_t priority, PackExecutionCallback cb):
              tensors_(tensors), 
              num_workers_(tensors.size()), priority(priority), cb_(std::move(cb)) {}

  bool IncrementReceivedCount();

  const std::vector<TensorPackElement> YieldTensorPackElements();

  int32_t get_tensor_id() const {return tensors_[0][0].tensor_id;}

  void on_finish() {cb_();}

  std::string to_string() const;

  int32_t priority;

private:
  int32_t num_workers_ = 0;
  int32_t current_step_ = 0;
  int32_t count_finished_ = 0;
  PackExecutionCallback cb_;
  std::vector<std::vector<TensorPackElement>> tensors_;
};

class SmallTensorPack {
public:
  struct compare_tp {
    bool operator()(const SmallTensorPack& l, const SmallTensorPack& r) {
      return l.priority > r.priority;
    }
  };

  SmallTensorPack() {};

  SmallTensorPack(const std::vector<TensorPackElement>& tensors, 
            int32_t tensor_priority, PackExecutionCallback callback, int32_t assigned_server):
              tensors_(tensors), 
              num_workers_(tensors.size()), 
              priority(tensor_priority), 
              cb_(std::move(callback)),
              assigned_server_(assigned_server) {}

  void SetFirstWorker(int32_t worker_id);

  int32_t get_first_worker() const {return worker_offset_;}

  const TensorPackElement YieldTensorPackElements();

  int32_t get_tensor_id() const {return tensors_[0].tensor_id;}

  int32_t get_assigned_server() const {return assigned_server_;}

  void on_finish() {cb_();}

  std::string to_string() const;

  int32_t priority;

protected:
  int32_t num_workers_ = 0;
  int32_t current_step_ = 0;
  int32_t worker_offset_ = 0;
  int32_t assigned_server_ = -1;
  PackExecutionCallback cb_;

private:
  std::vector<TensorPackElement> tensors_;
};

class PackExecutor {
public:
  PackExecutor(TensorManager& tensor_manager): 
                tensor_manager_(tensor_manager) {}

  Status SignalPushFinished();

  void Post(TensorPack pack);

  bool IsExecuting() const {return executing_;}

private:
  std::vector<Response> 
    ConstructResponse_(const std::vector<TensorPackElement>& tensors) const;

  bool ExecutePackInQueue_();

  std::priority_queue<TensorPack, std::deque<TensorPack>, TensorPack::compare_tp> 
    tp_queue_;
  
  bool executing_ = false;
  TensorPack executing_tp_;
  TensorManager& tensor_manager_;
};

class SmallTensorExecutor {
public:
  SmallTensorExecutor(TensorManager& tensor_manager):
                        tensor_manager_(tensor_manager) {}

  void Finalize();

  Status SignalPushFinished(int32_t tensor_id);

  void Post(SmallTensorPack pack);

private:
  Response
  ConstructResponse_(const TensorPackElement& tensor) const;

  bool ExecutePackInQueue_();

  inline void CheckFinalized();

  int32_t world_size_ = 0;
  
  std::priority_queue<SmallTensorPack, std::deque<SmallTensorPack>, SmallTensorPack::compare_tp> 
    tp_queue_;
  
  std::vector<SmallTensorPack> executing_tps_;
  std::unordered_set<int32_t> idle_servers;
  std::unordered_map<int32_t, int32_t> tensor_id_to_server_dict_;
  TensorManager& tensor_manager_;
  bool finalized_ = false;
};

class Controller {
public:
  Controller(): pack_executor_(tensor_manager_), 
                small_tensor_executor_(tensor_manager_) {}

  Controller(const Controller&) = delete;
  // Functions must be overridden by concrete controller
  virtual void Initialize() = 0;

  void LaunchBackGroundThread();

  int32_t get_rank() const {return rank_;};

  int32_t get_world_size() const {return world_size_;}

  // Interface to bytecore

  void PostTensor(std::vector<Tensor>& tensors, int32_t priority, int32_t assigned_server = -1);

  void SignalPartitionFinished(int32_t tensor_id, int32_t partition_id);

  void Log(const std::string str);

protected:
  // The main function called during each tick, includes the entire workflow
  // of the scheduler.
  //
  // The scheduler follows a master-worker paradigm. Rank zero acts
  // as the master (the "coordinator"), whereas all other ranks are simply
  // workers. Workers will communicate with coordinator to agree on what
  // tensors to be pushed. The communication performs as following:
  //
  //      a) The workers send Requests to the coordinator, indicating the 
  //      operations that they finished in the last tick. The coordinator
  //      receives the Requests from the workers
  //
  //      b) The coordinator processes the requests. If a tensor is ready or
  //      a push op is finished on every worker, it sends a response to workers.
  //
  //      e) The workers listen for Response messages, processing each one by
  //      doing the required operation. 
  void RunLoopOnce_();

  void RunMainLoop_();

  std::thread bg_thread;

  bool inited_ = false;
  // For rank 0 to receive other ranks' ready tensors.
  virtual std::vector<Request> RecvRequests_() = 0;

  // For other ranks to send their ready tensors to rank 0
  virtual void SendRequests_(const RequestList& request_list) = 0;

  virtual std::vector<Response> RecvResponses_() = 0;

  virtual std::vector<Response> 
    SendResponses_(const std::vector<ResponseList>& response_list) = 0;

  void ConstructTensorPacks_(int32_t tensor_id, int32_t num_partitions, 
                              int32_t priority, int32_t assigned_server);

  void DeleteFromTPTable(int32_t tensor_id);

  // Store the Request, and return whether the total count of
  // Requests for that tensor is now equal to the HOROVOD size (and thus we are
  // ready to reduce the tensor).
  bool IncrementTensorCount_(const Request& msg);

  // Only exists on coordinator node, processes each received requests and 
  // check if a tensor is ready
  void ProcessRequests_(const std::vector<Request>& recvd_requests);

  // Exists on worker side, processes the responses received from coordinator.
  void ProcessResponses_(const std::vector<Response>& recvd_responses);

  int rank_ = 0;
  int world_size_ = 1;
  bool is_coordinator_ = false;

  mutable std::mutex mutex_;

  TensorManager tensor_manager_;

  // The following only exists on the coordinator node (rank zero).
  // Maintains the ready count for each tensor.
  std::unordered_map<int32_t, int32_t> ready_table_;

  std::unordered_map<int32_t, std::vector<TensorPack>> tp_table_;

  std::unordered_map<int32_t, SmallTensorPack> stp_table_;

  std::unordered_map<int32_t, int32_t> tp_count_table_;

  PackExecutor pack_executor_;

  SmallTensorExecutor small_tensor_executor_;

};

} // namespace common
} // namespace proposed

#endif // PROPOSED_CONTROL_MANAGER_H
