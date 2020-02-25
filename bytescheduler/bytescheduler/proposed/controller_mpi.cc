#include <iostream>
#include "controller_mpi.h"

#define DEBUG_LOGGING true
#define DEBUG_PRINT(STR) if (DEBUG_LOGGING) Log(STR);

namespace proposed{
namespace common {

void MPIController::Initialize() {
  assert(inited_ == false);
  int32_t provided;
  MPI_Init_thread(NULL, NULL, MPI_THREAD_FUNNELED, &provided);
  assert(provided == MPI_THREAD_FUNNELED);
  MPI_Comm_size(comm_, &world_size_);
  MPI_Comm_rank(comm_, &rank_);

  tensor_manager_.SetRank(rank_);
  tensor_manager_.SetWorldSize(world_size_);

  if (rank_ == 0) is_coordinator_ = true;
  Log("MPI initialized. World size: " + std::to_string(world_size_));
  inited_ = true;
  tensor_manager_.Finalize();
}

std::vector<Request> MPIController::RecvRequests_() {
  std::vector<Request> recvd_reqs;
  assert(is_coordinator_ == true); // must be called by coordinator

  // Request of local machine
  RequestList local_reqs = tensor_manager_.ResetRequestList();
  for(auto&& req: local_reqs.requests()) {
    recvd_reqs.emplace_back(req);
  }
  // Get request of other machines
  // 1. Get the length of messages for each rank

  int32_t counts[world_size_];
  int32_t my_count = 0;
  MPI_Gather(&my_count, 1, MPI_INT32_T, counts, 
                        1, MPI_INT32_T, 0, comm_);

  // 2. Get the actual messages
  int32_t displs[world_size_];
  int32_t rcounts[world_size_];
  int32_t displ_counter = 0;
  for(int32_t i=0; i<world_size_; i++) {
    displs[i] = displ_counter;
    rcounts[i] = counts[i];
    displ_counter += counts[i];
  }
  uint8_t recv_buffer[displ_counter];
  MPI_Gatherv(NULL, 0, MPI_UNSIGNED_CHAR, recv_buffer, rcounts, 
              displs, MPI_UNSIGNED_CHAR, 0, comm_);
  
  // 3. Parse the requests
  for(int32_t worker_rank =1; worker_rank<world_size_; worker_rank++) {
    RequestList req_list;
    RequestList::ParseFromBytes(req_list, recv_buffer + displs[worker_rank]);
    for(auto&& req: req_list.requests()) {
      recvd_reqs.emplace_back(req);
    }
  }
  return recvd_reqs;
}

void MPIController::SendRequests_(const RequestList& request_list) {
  assert(is_coordinator_ == false); // must be called by worker

  // 1. Send the messsage size to rank 0 
  int32_t msg_size = request_list.size_in_serialized_bytes();
  MPI_Gather(&msg_size, 1, MPI_INT32_T, NULL, 0, NULL, 0, comm_);

  // 2. Send actual message to rank 0
  uint8_t send_buffer[msg_size];
  RequestList::SerializeToBytes(request_list, send_buffer);
  MPI_Gatherv(send_buffer, msg_size, MPI_UNSIGNED_CHAR, 
              NULL, NULL, NULL, NULL, 0, comm_);
}

std::vector<Response> MPIController::RecvResponses_() {
  std::vector<Response> recvd_responses;
  assert(is_coordinator_ == false); // must be called by worker

  // Get response from root
  // 1. Get the length of messages for me
  int32_t msg_size;
  MPI_Scatter(NULL, 0, NULL, &msg_size, 1, MPI_INT32_T, 0, comm_);

  // 2. Get the actual messages
  uint8_t recv_buffer[msg_size];
  MPI_Scatterv(NULL, NULL, NULL, NULL, recv_buffer, 
                msg_size, MPI_UNSIGNED_CHAR, 0, comm_);
  
  // 3. Parse received responses
  ResponseList res_list;
  ResponseList::ParseFromBytes(res_list, recv_buffer);
  for(auto&& res: res_list.responses()) {
    recvd_responses.emplace_back(res);
  }
  return recvd_responses;
}

std::vector<Response> 
MPIController::SendResponses_(const std::vector<ResponseList>& response_lists) {
  assert(is_coordinator_ == true); // must be called by coordinator

  // Response of local machine
  std::vector<Response> my_response;
  for(auto&& res: response_lists[rank_].responses()) {
    my_response.emplace_back(res);
  }

  // Send responses to other machines
  // 1. Send the length of messages to each rank
  int32_t counts[world_size_];
  counts[rank_] = 0;
  for(int worker_id = 1; worker_id < world_size_; worker_id ++) {
    counts[worker_id] = response_lists[worker_id].size_in_serialized_bytes();
  }
  int32_t my_count;
  MPI_Scatter(counts, 1, MPI_INT32_T, &my_count, 1, MPI_INT32_T, 0, comm_);

  // 2. Send the actual messages
  int32_t displs[world_size_];
  int32_t scounts[world_size_];
  int32_t displ_counter = 0;
  for(int32_t i=0; i<world_size_; i++) {
    displs[i] = displ_counter;
    scounts[i] = counts[i];
    displ_counter += counts[i];
  }
  uint8_t send_buffer[displ_counter];
  for(int worker_id=1; worker_id < world_size_; worker_id++) {
    ResponseList::SerializeToBytes(response_lists[worker_id], 
                                    send_buffer+displs[worker_id]);
  }
  MPI_Scatterv(send_buffer, scounts, displs, MPI_UNSIGNED_CHAR, 
                NULL, 0, MPI_UNSIGNED_CHAR, 0, comm_);
  
  return my_response;
}



} // namespace common
} // namespace proposed