#ifndef PROPOSED_MPI_CONTROL_MANAGER_H
#define PROPOSED_MPI_CONTROL_MANAGER_H

#include <mpi.h>
#include "controller.h"

namespace proposed {
namespace common {

class MPIController: public Controller {
public:
  MPIController() = default;

  MPIController(const MPIController&) = delete;

  void Initialize() override;

protected:
  // For rank 0 to receive other ranks' ready tensors.
  std::vector<Request> RecvRequests_() override;

  // For other ranks to send their ready tensors to rank 0
  void SendRequests_(const RequestList& request_list) override;

  std::vector<Response> RecvResponses_() override;

  std::vector<Response> 
    SendResponses_(const std::vector<ResponseList>& response_list) override;
    
private:

  MPI_Comm comm_ = MPI_COMM_WORLD;
};

} // namespace common
} // namespace proposed

#endif // PROPOSED_MPI_CONTROL_MANAGER_H
