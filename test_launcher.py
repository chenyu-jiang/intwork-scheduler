import proposed
import queue

proposed.init()
squeue = queue.Queue()

print("[rank {}] world size: {}".format(proposed.get_rank(), 
                                            proposed.get_world_size()))

def build_cb(layer_id, partition_id):
    return lambda: squeue.put((layer_id, partition_id))

layer_num = 5
partition_per_layer = 16

cbs = []
for layer_id in range(layer_num):
    layer_cbs = []
    layer_id_copy = layer_id
    for partition_id in range(partition_per_layer):
        layer_cbs.append(build_cb(layer_id, partition_id))
    cbs.append(layer_cbs)

for rev_layer_id in range(layer_num):
    layer_id = layer_num - rev_layer_id - 1
    proposed.post_tensor(layer_id, cbs[layer_id], layer_id)

for _ in range(layer_num*partition_per_layer):
    layer_id, partition_id = squeue.get()
    print("[rank {}]: Layer {} Partition {} finished.".format(proposed.get_rank(), layer_id, partition_id))
    proposed.proposed_signal_partition_finished(layer_id, partition_id)

print("[rank {}] Finished.".format(proposed.get_rank()))

