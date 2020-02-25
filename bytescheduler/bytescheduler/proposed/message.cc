// Copyright 2016 The TensorFlow Authors. All Rights Reserved.
// Modifications Chenyu Jiang
// Modifications copyright (C) 2019 Uber Technologies, Inc.
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

#include "message.h"

#include <iostream>

namespace proposed {
namespace common {

// Request =====================================================================

// Size of (type + rank + layer_id + partition_id)
const int32_t Request::SerializedRequestSize = sizeof(uint8_t) + 
                                              sizeof(int32_t)*3;

const std::string& Request::RequestType_Name(RequestType value) {
  switch (value) {
    case RequestType::TENSOR_READY:
      static const std::string layer_ready("ALLREDUCE");
      return layer_ready;
    case RequestType::PARTITION_FINISHED:
      static const std::string partition_finished("ALLGATHER");
      return partition_finished;
    default:
      static const std::string unknown("<unknown>");
      return unknown;
  }
}

Request::RequestType Request::request_type() const {
  return request_type_;
}

void Request::set_request_type(RequestType value) { request_type_ = value; }

int32_t Request::tensor_id() const { return tensor_id_; }

void Request::set_tensor_id(int32_t value) { tensor_id_ = value;}

int32_t Request::partition_id() const { return partition_id_; }

void Request::set_partition_id(int32_t value) { partition_id_ = value;}

void Request::set_request_rank(int32_t value) { request_rank_ = value;}

int32_t Request::request_rank() const { return request_rank_;}

void Request::ParseFromBytes(Request& request, const uint8_t* input) {
  // type
  request.set_request_type(static_cast<RequestType>(input[0]));
  int offset = 1;
  // rank
  int32_t request_rank = 0;
  for(int byte_index=0; byte_index<sizeof(int32_t); byte_index++) {
    request_rank += input[offset+byte_index] << byte_index * 8;
  }
  request.set_request_rank(request_rank);
  offset += sizeof(int32_t);
  // layer ID 
  int32_t layer_id = 0;
  for(int byte_index=0; byte_index<sizeof(int32_t); byte_index++) {
    layer_id += input[offset+byte_index] << byte_index * 8;
  }
  request.set_tensor_id(layer_id);
  offset += sizeof(int32_t);
  // partition id
  int32_t partition_id = 0;
  for(int byte_index=0; byte_index<sizeof(int32_t); byte_index++) {
    partition_id += input[offset+byte_index] << byte_index * 8;
  }
  request.set_partition_id(partition_id);
}

void Request::SerializeToBytes(const Request& request, uint8_t* output){
  // type
  output[0] = request.request_type();
  int offset = 1;
  // rank
  for(int byte_index=0; byte_index<sizeof(int32_t); byte_index++) {
    output[offset+byte_index] = request.request_rank() >> byte_index * 8;
  }
  offset += sizeof(int32_t);
  // layer ID 
  for(int byte_index=0; byte_index<sizeof(int32_t); byte_index++) {
    output[offset+byte_index] = (request.tensor_id() >> byte_index * 8);
  }
  offset += sizeof(int32_t);
  // partition id
  for(int byte_index=0; byte_index<sizeof(int32_t); byte_index++) {
    output[offset+byte_index] = (request.partition_id() >> byte_index * 8);
  }
}

void Request::SerializeToString(const Request& request,
                                std::string& output) {
  output = "Type:" + RequestType_Name(request.request_type()) + 
            ", Rank:" + std::to_string(request.request_rank()) + 
            ", LayerID:" + std::to_string(request.tensor_id()) + 
            ", PartitionID:" + std::to_string(request.partition_id());
}

// RequestList =================================================================

const std::vector<Request>& RequestList::requests() const {
  return requests_;
}

void RequestList::set_requests(const std::vector<Request>& value) {
  requests_ = value;
}

bool RequestList::shutdown() const { return shutdown_; }

void RequestList::set_shutdown(bool value) { shutdown_ = value; }

void RequestList::add_request(const Request& value) {
  requests_.push_back(value);
}

void RequestList::emplace_request(Request&& value) {
  requests_.emplace_back(value);
}

// size + requests
int32_t RequestList::size_in_serialized_bytes() const {
  return sizeof(int32_t) + 
                  requests_.size() * Request::SerializedRequestSize;
}

int32_t RequestList::size() const {
  return requests_.size();
}

void RequestList::ParseFromBytes(RequestList& request_list,
                                 const uint8_t* input) {
  int offset = 0;
  // size
  int32_t size = 0;
  for(int byte_index=0; byte_index<sizeof(int32_t); byte_index++) {
    size += input[offset+byte_index] << byte_index * 8;
  }
  offset += sizeof(int32_t);
  // requests
  for(int request_id=0; request_id<size; request_id++) {
    Request req = Request();
    Request::ParseFromBytes(req, input+offset);
    request_list.emplace_request(std::move(req));
    offset += Request::SerializedRequestSize;
  }
}

void RequestList::SerializeToBytes(const RequestList& request_list,
                                   uint8_t* output) {
  int offset = 0;
  // size
  for(int byte_index=0; byte_index<sizeof(int32_t); byte_index++) {
    output[offset+byte_index] = request_list.size() >> byte_index * 8;
  }
  offset += sizeof(int32_t);
  // requests
  for(const auto& request: request_list.requests()) {
    Request::SerializeToBytes(request, output+offset);
    offset += Request::SerializedRequestSize;
  }
}

// Response ====================================================================

// Size of (type + layer_id + partition_id)
const int32_t Response::SerializedResponseSize = sizeof(uint8_t) + 
                                                sizeof(int32_t)*2;

const std::string& Response::ResponseType_Name(ResponseType value) {
  switch (value) {
    case ResponseType::RELEASE:
      static const std::string release("RELEASE");
      return release;
    case ResponseType::ERROR:
      static const std::string error("ERROR");
      return error;
    default:
      static const std::string unknown("<unknown>");
      return unknown;
  }
}

Response::ResponseType Response::response_type() const {
  return response_type_;
}

void Response::set_response_type(ResponseType value) {
  response_type_ = value;
}

int32_t Response::tensor_id() const {return tensor_id_;}

void Response::set_tensor_id(int32_t value) {tensor_id_ = value;}

int32_t Response::partition_id() const {return partition_id_;}

void Response::set_partition_id(int32_t value) {partition_id_=value;}

int32_t Response::rank() const { return rank_; }

void Response::set_rank(int32_t value) { rank_ = value; }

void Response::ParseFromBytes(Response& response, const uint8_t* input) {
  // type
  response.set_response_type(static_cast<ResponseType>(input[0]));
  int offset = 1;
  // layer ID 
  int32_t layer_id = 0;
  for(int byte_index=0; byte_index<sizeof(int32_t); byte_index++) {
    layer_id += input[offset+byte_index] << byte_index * 8;
  }
  response.set_tensor_id(layer_id);
  offset += sizeof(int32_t);
  // partition id
  int32_t partition_id = 0;
  for(int byte_index=0; byte_index<sizeof(int32_t); byte_index++) {
    partition_id += input[offset+byte_index] << byte_index * 8;
  }
  response.set_partition_id(partition_id);
}

void Response::SerializeToBytes(const Response& response, uint8_t* output) {
  // type
  output[0] = response.response_type();
  int offset = 1;
  // layer ID 
  for(int byte_index=0; byte_index<sizeof(int32_t); byte_index++) {
    output[offset+byte_index] = (response.tensor_id() >> byte_index * 8);
  }
  offset += sizeof(int32_t);
  // partition id
  for(int byte_index=0; byte_index<sizeof(int32_t); byte_index++) {
    output[offset+byte_index] = (response.partition_id() >> byte_index * 8);
  }
}

void Response::SerializeToString(const Response& response,
                                 std::string& output) {
  output = "Type:" + ResponseType_Name(response.response_type()) +  
          ", LayerID:" + std::to_string(response.tensor_id()) + 
          ", PartitionID:" + std::to_string(response.partition_id());
}

// ResponseList ================================================================

const std::vector<Response>& ResponseList::responses() const {
  return responses_;
}

void ResponseList::set_responses(const std::vector<Response>& value) {
  responses_ = value;
}

bool ResponseList::shutdown() const { return shutdown_; }

void ResponseList::set_shutdown(bool value) { shutdown_ = value; }

void ResponseList::add_response(const Response& value) {
  responses_.push_back(value);
}

void ResponseList::emplace_response(Response&& value) {
  responses_.emplace_back(value);
}

// size + responses
int32_t ResponseList::size_in_serialized_bytes() const {
  return sizeof(int32_t) + 
                  responses_.size() * Response::SerializedResponseSize;
}

int32_t ResponseList::size() const {
  return responses_.size();
}

void ResponseList::ParseFromBytes(ResponseList& response_list,
                                 const uint8_t* input) {
  int offset = 0;
  // size
  int32_t size = 0;
  for(int byte_index=0; byte_index<sizeof(int32_t); byte_index++) {
    size += input[offset+byte_index] << byte_index * 8;
  }
  offset += sizeof(int32_t);
  // responses
  for(int response_id=0; response_id<size; response_id++) {
    Response res = Response();
    Response::ParseFromBytes(res, input+offset);
    response_list.emplace_response(std::move(res));
    offset += Response::SerializedResponseSize;
  }
}

void ResponseList::SerializeToBytes(const ResponseList& response_list,
                                   uint8_t* output) {
  int offset = 0;
  // size
  for(int byte_index=0; byte_index<sizeof(int32_t); byte_index++) {
    output[offset+byte_index] = response_list.size() >> byte_index * 8;
  }
  offset += sizeof(int32_t);
  // responses
  for(const auto& response: response_list.responses()) {
    Response::SerializeToBytes(response, output+offset);
    offset += Response::SerializedResponseSize;
  }
}

} // namespace common
} // namespace horovod
