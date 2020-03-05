// Copyright 2019 Uber Technologies, Inc. All Rights Reserved.
// Modifications copyright Microsoft
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

#include "tensor_manager.h"

#include <assert.h>

namespace proposed {
namespace common {

// TensorManager ===============================================================

void TensorManager::Finalize() {
  if (rank_ == -1 || world_size_ == 0) {
    assert( false && "Must register tensors before finalizing TensorManager.");
  }
  response_list_.resize(world_size_);
  finalized_ = true;
}

void TensorManager::PostTensor(std::vector<Tensor>& tensors, int32_t priority, int32_t assigned_server) {
  std::lock_guard<std::mutex> guard(mutex_);
  CheckFinalized();
  // std::cout << "At the start of PostTensor." << std::endl;
  // Add tensor into the tensor table
  if(tensors.size() == 0)
    assert(false && "Posted tensor with size 0.");
  // std::cout << "Before getting tensor_id." << std::endl;
  int32_t tensor_id = tensors[0].tensor_id;

  // std::cout << "Before checking dup key." << std::endl;
  if (tensor_table_.find(tensor_id) != tensor_table_.end()) {
    std::cout<< "Duplicate Key detected: " + std::to_string(tensor_id) << std::endl;
    assert(false && "Duplicate key in tensor table.");
  }

  tensor_table_.emplace(tensor_id, 
                        TensorTableElement(tensors, tensor_id, priority));
  assert(tensor_count_table_[tensor_id] == 0);
  tensor_count_table_[tensor_id] = tensors.size();
  
  // Check if is small tensor
  if(tensors.size() == 1) {
    // small tensor, register assigned_server
    if(assigned_server == -1) 
      assert(false && "Assigned server not specified for small tensor.");
    assigned_server_dict_[tensor_id] = assigned_server;
  }

  // formulate a layer ready request and push it into the queue
  request_list_.emplace_request(
    Request(rank_, Request::TENSOR_READY, tensor_id, -1));
  // std::cout << "Placed Request." << std::endl;
}

int32_t TensorManager::GetAssignedServer(int32_t tensor_id) const {
  std::lock_guard<std::mutex> guard(mutex_);
  if(assigned_server_dict_.find(tensor_id) == assigned_server_dict_.end())
    return -1;
  return assigned_server_dict_.at(tensor_id);
}

void TensorManager::DeleteTensorWithID(int32_t tensor_id) {
  tensor_table_.erase(tensor_id);
  // std::cout << "In TensorManager: check erased " + std::to_string(tensor_table_.find(tensor_id) == tensor_table_.end()) << std::endl;
  if(assigned_server_dict_.find(tensor_id) != assigned_server_dict_.end())
    assigned_server_dict_.erase(tensor_id);
}

void 
TensorManager::SignalPartitionFinished(int32_t tensor_id, int32_t partition_id) {
  std::lock_guard<std::mutex> guard(mutex_);
  CheckFinalized();
  request_list_.emplace_request(Request(rank_, Request::PARTITION_FINISHED, 
                          tensor_id, partition_id));
}

void TensorManager::ReleaseTensor(int32_t tensor_id, int32_t partition_id) {
  CheckFinalized();
  {
    std::lock_guard<std::mutex> guard(mutex_);
    // std::cout << "Releasing id " + std::to_string(tensor_id) + " p " + std::to_string(partition_id) + " at " + std::to_string(rank_) << std::endl;
    GetTensor(tensor_id, partition_id).ready_cb();
    tensor_count_table_[tensor_id] --;
    if (tensor_count_table_[tensor_id] == 0) {
      DeleteTensorWithID(tensor_id);
    }
  }
}

// Parse tensor from response and get corresponding tensor entry.
const Tensor& TensorManager::GetTensorFromResponse(const Response& response) {
  std::lock_guard<std::mutex> guard(mutex_);
  CheckFinalized();
  if (response.response_type() != Response::ERROR) {
    return GetTensor(response.tensor_id(), response.partition_id());
  } else {
    assert(false && "Received response with type ERROR.");
  }
}

// Get tensor entry given a tensor_id and partition_id
const Tensor&
TensorManager::GetTensor(const int32_t tensor_id, 
                          const int32_t partition_id) const {
  if (tensor_table_.find(tensor_id) == tensor_table_.end()) {
    assert(false && "GetTensor: TensorTable entry do not exist.");
  }
  const TensorTableElement& te = tensor_table_.at(tensor_id);

  if (partition_id >= te.size())
    assert(false && "Partition do not exist.");
  
  return te.tensors[partition_id];
}

int32_t TensorManager::GetNumPartitions(int32_t tensor_id) {
  std::lock_guard<std::mutex> guard(mutex_);
  if (tensor_table_.find(tensor_id) == tensor_table_.end()) {
    assert(false && "GetNumPartitions: TensorTable entry do not exist.");
  }
  return tensor_table_.at(tensor_id).size();
}

int32_t TensorManager::GetPriority(int32_t tensor_id) {
  std::lock_guard<std::mutex> guard(mutex_);
  if (tensor_table_.find(tensor_id) == tensor_table_.end()) {
    assert( false && "GetPriority: TensorTable entry do not exist.");
  }
  return tensor_table_.at(tensor_id).priority;
}

// Pop out all the messages from the queue
RequestList TensorManager::ResetRequestList() {
  std::lock_guard<std::mutex> guard(mutex_);
  CheckFinalized();
  RequestList tmp_req_list = request_list_;
  request_list_ = RequestList();
  return tmp_req_list;
}

// Pop out all the messages from the queue
std::vector<ResponseList> TensorManager::ResetResponseList() {
  std::lock_guard<std::mutex> guard(mutex_);
  CheckFinalized();
  std::vector<ResponseList> tmp_res_list = response_list_;
  response_list_ = std::vector<ResponseList>();
  response_list_.resize(world_size_);
  return tmp_res_list;
}

// Push a request to request list
void TensorManager::AddRequestToList(Request& message) {
  std::lock_guard<std::mutex> guard(mutex_);
  request_list_.emplace_request(std::move(message));
}

// Push a response to response list
void TensorManager::AddResponseToList(Response& message) {
  std::lock_guard<std::mutex> guard(mutex_);
  response_list_[message.rank()].emplace_response(std::move(message));
}



} // namespace common
} // namespace horovod
