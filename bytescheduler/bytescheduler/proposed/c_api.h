#include "controller_mpi.h"

namespace proposed{
namespace common {

static MPIController controller_;

// C functions exposed as API
extern "C" {

typedef void (*C_ReadyCallback)();

int32_t proposed_init();

int32_t proposed_get_rank();

int32_t proposed_get_world_size();

int32_t proposed_post_tensor(int32_t tensor_id, 
                             int32_t num_partitions, 
                             C_ReadyCallback* cb,
                             int32_t priority);

int32_t proposed_signal_partition_finished(int32_t tensor_id, 
                                           int32_t partition_id);

} // extern "C"

} // namespace common
} // namespace proposed