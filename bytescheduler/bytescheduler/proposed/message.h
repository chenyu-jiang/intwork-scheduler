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

#ifndef PROPOSED_MESSAGE_H
#define PROPOSED_MESSAGE_H

#include <string>
#include <vector>
#include <unordered_map>

namespace proposed {
namespace common {

// A Request is a message sent from a rank greater than zero to the
// coordinator (rank zero), informing the coordinator that a specific 
// operation is ready at the rank.
class Request {
public:
  enum RequestType: uint8_t {
    TENSOR_READY= 0, PARTITION_FINISHED = 1
  };

  Request() {};

  Request(int32_t rank, Request::RequestType type, 
            int32_t tensor_id, int32_t partition_id) {
    request_rank_ = rank;
    request_type_ = type;
    tensor_id_ = tensor_id;
    partition_id_ = partition_id;
  }

  // SerialiedRequestSize holds the length of the serialized request message 
  // in number of bytes. 
  static const int32_t SerializedRequestSize;

  static const std::string& RequestType_Name(RequestType value);

  // The request rank is necessary to create a consistent ordering of results,
  // for example in the allgather where the order of outputs should be sorted
  // by rank.
  int32_t request_rank() const;

  void set_request_rank(int32_t value);

  RequestType request_type() const;

  void set_request_type(RequestType value);

  int32_t tensor_id() const;

  void set_tensor_id(const int32_t value);

  int32_t partition_id() const;

  void set_partition_id(const int32_t value);

  static void ParseFromBytes(Request& request, const uint8_t* input);

  static void SerializeToBytes(const Request& request, uint8_t* output);

  static void SerializeToString(const Request& request, std::string& output);

private:
  int32_t request_rank_ = 0;
  RequestType request_type_ = RequestType::TENSOR_READY;
  int32_t tensor_id_ = 0;
  int32_t partition_id_ = 0;
};

class RequestList {
public:
  const std::vector<Request>& requests() const;

  void set_requests(const std::vector<Request>& value);

  void add_request(const Request& value);

  void emplace_request(Request&& value);

  bool shutdown() const;

  void set_shutdown(bool value);

  int32_t size_in_serialized_bytes() const;

  int32_t size() const;

  void operator += (const RequestList& other) {
    
  }

  static void ParseFromBytes(RequestList& request_list,
                             const uint8_t* input);

  static void SerializeToBytes(const RequestList& request_list,
                               uint8_t* output);

private:
  std::vector<Request> requests_;
  bool shutdown_ = false;
};

// A Response is a message sent from the coordinator (rank zero) to a rank
// greater than zero, informing the rank of an operation should be performed
// now. If the operation requested would result in an error (for example, due
// to a type or shape mismatch), then the Response can contain an error and
// an error message instead.
class Response {
public:
  enum ResponseType: uint8_t{
    RELEASE=0, ERROR = 1
  };

  Response() = default;
  Response(ResponseType type, int32_t rank, 
            int32_t tensor_id, int32_t partition_id):
    response_type_(type), rank_(rank), 
    tensor_id_(tensor_id), partition_id_(partition_id) {}

  // SerialiedResponseSize holds the length of the serialized response message 
  // in number of bytes. 
  static const int32_t SerializedResponseSize;

  static const std::string& ResponseType_Name(ResponseType value);

  ResponseType response_type() const;

  void set_response_type(ResponseType value);

  int32_t tensor_id() const;

  void set_tensor_id(int32_t value);

  int32_t partition_id() const;

  void set_partition_id(int32_t value);

  int32_t rank() const;

  void set_rank(int32_t value);

  static void ParseFromBytes(Response& response, const uint8_t* input);

  static void SerializeToBytes(const Response& response, uint8_t* output);

  static void SerializeToString(const Response& response,
                                std::string& output);

private:
  ResponseType response_type_ = ResponseType::RELEASE;
  int32_t rank_ = 0;
  int32_t tensor_id_ = 0;
  int32_t partition_id_ = 0;
};

class ResponseList {
public:
  const std::vector<Response>& responses() const;

  void set_responses(const std::vector<Response>& value);

  void add_response(const Response& value);

  void emplace_response(Response&& value);

  bool shutdown() const;

  void set_shutdown(bool value);

  int32_t size_in_serialized_bytes() const;

  int32_t size() const;

  static void ParseFromBytes(ResponseList& response_list,
                             const uint8_t* input);

  static void SerializeToBytes(const ResponseList& response_list,
                               uint8_t* output);

private:
  std::vector<Response> responses_;
  bool shutdown_ = false;
};

} // namespace common
} // namespace proposed

#endif // PROPOSED_MESSAGE_H
