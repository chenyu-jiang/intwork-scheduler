#include "c_api.h"

namespace proposed{
namespace common {

// C functions exposed as API
extern "C" {

int32_t proposed_init() {
    controller_.Initialize();
    controller_.LaunchBackGroundThread();
    return 0;
}

int32_t proposed_get_rank() {
    return controller_.get_rank();
}

int32_t proposed_get_world_size() {
    return controller_.get_world_size();
}

int32_t proposed_post_tensor(int32_t tensor_id, 
                             int32_t num_partitions, 
                             C_ReadyCallback* cb, 
                             int32_t priority,
                             int32_t assigned_server) {
    // Convert C style API call to C++ style calls
    std::vector<Tensor> ts;
    for(int i=0;i<num_partitions; i++) {
        C_ReadyCallback this_cb = *(cb + i); 
        ts.emplace_back(Tensor(tensor_id, i, [this_cb]() { this_cb();}));
    }
    controller_.PostTensor(ts, priority, assigned_server);
    return 0;
}

int32_t proposed_signal_partition_finished(int32_t tensor_id, 
                                           int32_t partition_id) {
    controller_.SignalPartitionFinished(tensor_id, partition_id);
    return 0;
}

} // extern "C"

} // namespace common
} // namespace proposed