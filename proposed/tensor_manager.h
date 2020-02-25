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

#ifndef PROPOSED_TENSOR_QUEUE_H
#define PROPOSED_TENSOR_QUEUE_H

#include <iostream>
#include <mutex>
#include <queue>

#include "common.h"
#include "message.h"

namespace proposed {
namespace common {

class TensorManager {
public:
  TensorManager() = default;
  TensorManager(int32_t rank, int32_t world_size):
    rank_(rank), world_size_(world_size) {};
  TensorManager(const TensorManager&) = delete;

  void Finalize();

  void SetRank(int32_t value) {rank_ = value;}

  void SetWorldSize(int32_t value) {world_size_ = value;}

  int32_t GetNumPartitions(int32_t tensor_id);

  int32_t GetPriority(int32_t tensor_id);

  void PostTensor(std::vector<Tensor>& t, int32_t priority);

  void DeleteTensorWithID(int32_t tensor_id);

  void SignalPartitionFinished(int32_t layer_id, int32_t partition_id);

  void ReleaseTensor(int32_t layer_id, int32_t partition_id);

  const Tensor& GetTensorFromResponse(const Response& response);

  const Tensor& GetTensor(const int32_t layer_id,
                          const int32_t partition_id) const;

  RequestList ResetRequestList();

  std::vector<ResponseList> ResetResponseList();

  void AddRequestToList(Request& message);

  void AddResponseToList(Response& message, int32_t rank);

protected:
  void CheckFinalized() {
    if (!finalized_) throw std::runtime_error(
      "TensorManager must be finalized before starting to operate.");
  }

  bool finalized_ = false;
  
  int32_t rank_ = -1;

  int32_t world_size_ = 0;

  TensorTable tensor_table_;

  // Queue of requests waiting to be sent to the coordinator node.
  RequestList request_list_;

  std::vector<ResponseList> response_list_;

  mutable std::mutex mutex_;
};

} // namespace common
} // namespace proposed

#endif // PROPOSED_TENSOR_QUEUE_H
