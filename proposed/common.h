// Copyright 2018 Uber Technologies, Inc. All Rights Reserved.
// Modifications copyright (C) 2019 Intel Corporation
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

#ifndef PROPOSED_COMMON_H
#define PROPOSED_COMMON_H

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

#include "message.h"

namespace proposed {
namespace common {

enum StatusType { OK, UNKNOWN_ERROR, PRECONDITION_ERROR, ABORTED, 
                  INVALID_ARGUMENT, IN_PROGRESS };

class Status {
public:
  Status();
  static Status OK();
  static Status UnknownError(std::string message);
  static Status PreconditionError(std::string message);
  static Status Aborted(std::string message);
  static Status InvalidArgument(std::string message);
  static Status InProgress();
  bool ok() const;
  bool in_progress() const;
  StatusType type() const;
  const std::string& reason() const;

private:
  StatusType type_ = StatusType::OK;
  std::string reason_ = "";
  Status(StatusType type, std::string reason);
};

// Common error status
const Status NOT_INITIALIZED_ERROR = Status::PreconditionError(
    "Horovod has not been initialized; use hvd.init().");

const Status DO_NOT_EXIST_ERROR = Status::PreconditionError(
    "Tensor do not exist.");

const Status SHUT_DOWN_ERROR = Status::UnknownError(
    "Horovod has been shut down. This was caused by an exception on one of the "
    "ranks or an attempt to allreduce, allgather or broadcast a tensor after "
    "one of the ranks finished execution. If the shutdown was caused by an "
    "exception, you should see the exception in the log before the first "
    "shutdown message.");

const Status DUPLICATE_NAME_ERROR = Status::InvalidArgument(
    "Requested to allreduce, allgather, or broadcast a tensor with the same "
    "name as another tensor that is currently being processed.  If you want "
    "to request another tensor, use a different tensor name.");

using ReadyCallback = std::function<void()>;

class Tensor {
public:
  Tensor();

  Tensor(int32_t tensor_id, int32_t partition_id, ReadyCallback ready_cb): 
    tensor_id(tensor_id), partition_id(partition_id), ready_cb(ready_cb) {}

  Tensor(const Tensor& other) {
    tensor_id = other.tensor_id;
    partition_id = other.partition_id;
    ready_cb = other.ready_cb;
  }

  std::string to_string() const {
    return "l: " + std::to_string(tensor_id) + 
            ", p: " + std::to_string(partition_id) + ";";
  }

  int32_t tensor_id = 0;
  int32_t partition_id = 0;
  ReadyCallback ready_cb;
};

using Int32Pair = std::pair<int32_t, int32_t>;

inline Int32Pair make_key(int32_t v1, int32_t v2) {
  return std::make_pair(v1, v2);
}

struct hash_pair { 
    template <class T1, class T2> 
    size_t operator()(const std::pair<T1, T2>& p) const
    { 
        auto hash1 = std::hash<T1>{}(p.first); 
        auto hash2 = std::hash<T2>{}(p.second); 
        return hash1 ^ hash2; 
    } 
};

class TensorTableElement {
public:
  TensorTableElement(std::vector<Tensor>& tensors, 
                      int32_t tensor_id, int32_t priority):
    tensors(tensors), tensor_id(tensor_id), priority(priority) {}

  int32_t size() const {return tensors.size();}
  
  std::vector<Tensor> tensors;
  int32_t tensor_id = 0;
  int32_t priority = 0;
};

using TensorTable = std::unordered_map<int32_t, TensorTableElement>;

} // namespace common
} // namespace proposed

#endif // PROPOSED_COMMON_H
