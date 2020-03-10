import bytescheduler.proposed as proposed
import queue
import random

proposed.init()
squeue = queue.Queue()

print("[rank {}] world size: {}".format(proposed.get_rank(), 
                                            proposed.get_world_size()))

def build_cb(layer_id, partition_id):
    return lambda: squeue.put((layer_id, partition_id))

layer_num = 5
partition_per_layer = 16

num_small_tensor_per_layer = 5

cbs = []
for layer_id in range(layer_num):
    layer_cbs = []
    layer_id_copy = layer_id
    for partition_id in range(partition_per_layer):
        layer_cbs.append(build_cb(layer_id, partition_id))
    cbs.append(layer_cbs)

small_cbs = []
for layer_id in range(layer_num):
    for st_id in range(num_small_tensor_per_layer):
        small_cbs.append([build_cb((layer_id+1)*1000+st_id, 0)])

for rev_layer_id in range(layer_num):
    layer_id = layer_num - rev_layer_id - 1
    proposed.post_tensor(layer_id, cbs[layer_id], layer_id)
    for st_id in range(num_small_tensor_per_layer):
        proposed.post_tensor((layer_id+1)*1000+st_id, small_cbs[layer_id*num_small_tensor_per_layer+st_id], layer_id, 1)

for _ in range(layer_num*(partition_per_layer+num_small_tensor_per_layer)):
    layer_id, partition_id = squeue.get()
    if layer_id >= 1000:
        print("[rank {}]: Layer {} Partition {} finished.".format(proposed.get_rank(), layer_id, partition_id))
    proposed.proposed_signal_partition_finished(layer_id, partition_id)

print("[rank {}] Finished.".format(proposed.get_rank()))

