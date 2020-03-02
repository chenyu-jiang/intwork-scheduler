"""
A Python wrapper for the underlying C api.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ctypes
import collections
import sys

class ProposedWrapper(object):
    def __init__(self, path_to_shard_lib):
        self.COMM_CTYPES = ctypes.CDLL(path_to_shard_lib)
        self._callbacks = {}
        self._inited = False
    
    def init(self):
        if self._inited:
            return
        status = self.COMM_CTYPES.proposed_init()
        if status != 0:
            raise RuntimeError("Failed to initialize proposed scheduler.")
        self._inited = True
    
    def get_rank(self):
        if not self._inited:
            return -1
        return self.COMM_CTYPES.proposed_get_rank()
    
    def get_world_size(self):
        if not self._inited:
            return -1
        return self.COMM_CTYPES.proposed_get_world_size()
    
    def post_tensor(self, tensor_id, finish_cbs, priority, assigned_server = -1):
        print("[{}] post_tensor is called with tensor_id {}.".format(self.get_rank(), tensor_id))
        if not self._inited:
            raise RuntimeError("Must call init() before posting tensor to proposed scheduler.")
        num_partitions = len(finish_cbs)
        CB_ARRAY_TYPE = ctypes.CFUNCTYPE(None) * num_partitions
        cbs = []
        
        if num_partitions == 1:
            # small tensor, must specify assigned_server
            if(assigned_server == -1):
                raise RuntimeError("Must specify assigned server for small tensors.")
            if(assigned_server >= self.get_world_size()):
                raise RuntimeError("Specified assigned server larger than world size.")

        for pid in range(num_partitions):
            finish_cb = finish_cbs[pid]
            _CB_TYPE = ctypes.CFUNCTYPE(None)
            finish_cb_ctypes = _CB_TYPE(lambda f=finish_cb: f())
            self._callbacks[(tensor_id, pid)] = finish_cb_ctypes
            cbs.append(finish_cb_ctypes)

        status = self.COMM_CTYPES.proposed_post_tensor(
            ctypes.c_int32(tensor_id),
            ctypes.c_int32(num_partitions),
            CB_ARRAY_TYPE(*cbs),
            ctypes.c_int32(priority),
            ctypes.c_int32(assigned_server)
            )

        if status != 0:
            raise RuntimeError("Failed to post tensor to proposed scheduler.")

    def proposed_signal_partition_finished(self, tensor_id, partition_id):
        print("[{}] signal_partition_finished called.".format(self.get_rank()))
        if not self._inited:
            raise RuntimeError("Must call init() before signaling partition finished.")
        status = self.COMM_CTYPES.proposed_signal_partition_finished(
                    ctypes.c_int32(tensor_id), ctypes.c_int32(partition_id))
        if status != 0:
            raise RuntimeError("Failed to signal partition finished.")
