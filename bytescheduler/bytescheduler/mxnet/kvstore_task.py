from __future__ import absolute_import

import os
from ctypes import CDLL, RTLD_GLOBAL, CFUNCTYPE, c_int, byref, c_void_p, c_char_p

from mxnet.base import check_call, NDArrayHandle
from mxnet.ndarray import NDArray
from mxnet.ndarray import zeros
from ..common import get_ext_suffix
from ..common.bytetask import ByteTask

# Load c_lib.so
dll_path = os.path.join(os.path.dirname(__file__),
                        'c_lib' + get_ext_suffix())
BYTESCHEDULER_LIB = CDLL(dll_path, RTLD_GLOBAL)
callback_t = CFUNCTYPE(None, c_void_p)

barrier_tensors = {}


def get_barrier_tensor(key):
    global barrier_tensors
    if key not in barrier_tensors:
        barrier_tensors[key] = zeros(1)
    return barrier_tensors[key]


class KVStoreTask(ByteTask):
    def _additional_init(self):
        if self.op == "push_pull":
            if self.parent is not None:
                assert len(self._tensor) == self._num_workers
                self._push_tensor = []
                self._pull_tensor = []
                for tensor in self._tensor:
                    assert len(tensor) == 2
                    self._push_tensor.append(tensor[0])
                    self._pull_tensor.append(tensor[1])
            else:
                assert len(self._tensor) == 2
                self._push_tensor = self._tensor[0]
                self._pull_tensor = self._tensor[1]

            # unique key, assuming at most 10^6 tensors and each can be partitioned into at most 1000 partition
            self._barrier_key = str(self._partition_index + self.kwargs["key_index"]*1000 + 10**6)
            # used as a barrier for push_pull task
            self._barrier_tensor = get_barrier_tensor(self._barrier_key)
        elif self.op == "init":
            # worker 0 needs to init barrier tensor on PS
            self._barrier_key = str(self._partition_index + self.kwargs["key_index"]*1000 + 10**6)
            self._barrier_tensor = get_barrier_tensor(self._barrier_key)
            if self.parent is None:
                self._comm.init(self._barrier_key, self._barrier_tensor, server_assigned = self.kwargs["key_index"] % self._num_workers, is_barrier=True)
            else:
                self._comm.init(self._barrier_key, self._barrier_tensor, server_assigned = 0, is_barrier=True)


    def _post_communication(self, tensor):
        """Start send a tensor
        Args:
            tensor: a list of tensor to be init/push/pull.
        """
        if self.parent is None:
            if self.op == "init":
                if isinstance(tensor, (tuple, list)):
                    self._comm.init(self.name, tensor, server_assigned = [self.kwargs["key_index"] % self._num_workers]*len(tensor))
                else:
                    self._comm.init(self.name, tensor, server_assigned = self.kwargs["key_index"] % self._num_workers)
            elif self.op == "push":
                self._comm.push(self.name, tensor, -self.priority, is_small_tensor=True)
            elif self.op == "pull":
                self._comm.pull(self.name, out=tensor, priority=-self.priority - 1, ignore_sparse=self.kwargs["ignore_sparse"])
            elif self.op == "push_pull":
                assert len(tensor) == 2
                self._comm.push(self.name, tensor[0], -self.priority, is_small_tensor=True)
                # add an op to notify push completion
                self._push_completion_op_name = c_char_p(self.name.encode('ascii'))

                def push_completion_callback(on_complete):
                    # Call on_complete directly
                    check_call(BYTESCHEDULER_LIB.bytescheduler_mxnet_on_complete(
                        c_void_p(on_complete)))
                    # Called after push instead pull
                    self.notify_finish()

                # Avoid garbage collection
                self._push_completion_callback = callback_t(push_completion_callback)

                push_avatar = [t.handle for t in tensor[0]] if isinstance(tensor[0], (tuple, list)) else [tensor[0].handle]

                push_tensors_out = (NDArrayHandle * len(push_avatar))(*push_avatar)

                check_call(BYTESCHEDULER_LIB.bytescheduler_mxnet_op(
                    push_tensors_out, 0, push_tensors_out, len(push_avatar), self._push_completion_callback, 100000000-self.priority))

                self._comm.pull(self.name, out=tensor[1], priority=-self.priority-1, ignore_sparse=self.kwargs["ignore_sparse"])
            else:
                self._logger.error("ERROR: unexpected op type {}!".format(self.op))
        else:
            def map2pid(step_id):
                return (step_id + self._rank) % self._num_workers

            def tensor_key(step_id):
                return self.name + "_" + str(map2pid(step_id))

            def unmap(partition_id):
                return (partition_id + self._num_workers - self._rank) % self._num_workers

            if self.op == "init":
                # for init, rotate the tensors back to the normal order
                for partition_id in range(len(tensor)):
                    step_id = unmap(partition_id)
                    # print("BS: assigned server {} to {}.".format(partition_id, tensor_key(step_id)))
                    if isinstance(tensor[step_id], (tuple, list)):
                        self._comm.init(tensor_key(step_id), tensor[step_id], server_assigned = [partition_id]*len(tensor[step_id]))
                        # self._comm.init(tensor_key(step_id), tensor[step_id])
                    else:
                        self._comm.init(tensor_key(step_id), tensor[step_id], server_assigned = partition_id)
                        # self._comm.init(tensor_key(step_id), tensor[step_id])
            elif self.op == "push":
                for step_id in range(len(tensor)):
                    # print("Pushing {} with value {}.".format(tensor_key(step_id), tensor[step_id]))
                    self._comm.push(tensor_key(step_id), tensor[step_id], -self.priority - map2pid(step_id)*10)
            elif self.op == "pull":
                for step_id in range(len(tensor)):
                    self._comm.pull(tensor_key(step_id), out=tensor[step_id], priority=-self.priority - map2pid(step_id)*10 - 1, ignore_sparse=self.kwargs["ignore_sparse"])
            elif self.op == "push_pull":
                for step_id in range(len(tensor)):
                    assert len(tensor[step_id]) == 2
                    step_tensor = tensor[step_id]
                    # print("[{}] Pushing {} with priority {}.".format(self._rank, tensor_key(step_id), -self.priority - map2pid(step_id)*10))
                    self._comm.push(tensor_key(step_id), step_tensor[0], -self.priority - map2pid(step_id)*10)


                    push_avatar = [t.handle for t in step_tensor[0]] if isinstance(step_tensor[0], (tuple, list)) else [step_tensor[0].handle]
                    push_tensors_out = (NDArrayHandle * len(push_avatar))(*push_avatar)

                    if step_id == (len(tensor) - 1):
                        # add an op to notify push completion
                        self._push_completion_op_name = c_char_p(self.name.encode('ascii'))

                        def push_completion_callback(on_complete):
                            # Call on_complete directly
                            check_call(BYTESCHEDULER_LIB.bytescheduler_mxnet_on_complete(
                                c_void_p(on_complete)))
                            # Called after push instead pull
                            # print("{} finished.".format(self.desc))
                            self.notify_finish()

                        # Avoid garbage collection
                        self._push_completion_callback = callback_t(push_completion_callback)

                        check_call(BYTESCHEDULER_LIB.bytescheduler_mxnet_op(
                            push_tensors_out, 0, push_tensors_out, len(push_avatar), self._push_completion_callback, 100000000-self.priority))
                    else:
                        # add dependency between push ops
                        next_push_avatar = [t.handle for t in tensor[step_id+1][0]] if isinstance(tensor[step_id+1][0], (tuple, list)) else [tensor[step_id+1][0].handle]
                        next_push_avatar_out = (NDArrayHandle * len(next_push_avatar))(*next_push_avatar)
                        check_call(BYTESCHEDULER_LIB.bytescheduler_mxnet_barrier(
                                    push_tensors_out, len(push_avatar),
                                    next_push_avatar_out, len(next_push_avatar),
                                    100000000 - self.priority + step_id))
                    # print("[{}] Pulling {} with priority {}.".format(self._rank, tensor_key(step_id), -self.priority - map2pid(step_id)*10 -1))
                    self._comm.pull(tensor_key(step_id), out=step_tensor[1], priority=-self.priority - map2pid(step_id) - 1, ignore_sparse=self.kwargs["ignore_sparse"])
            else:
                self._logger.error("ERROR: unexpected op type {}!".format(self.op))


    def _do(self):
        """Let the start OP complete so that the real comm OP can run../."""
        if hasattr(self, "_on_complete"):
            # print("_do() has run in {}".format(self.desc))
            check_call(BYTESCHEDULER_LIB.bytescheduler_mxnet_on_complete(
                c_void_p(self._on_complete)))
        return

    def _prepare(self):
        """Post start barrier OP, start OP, comm OP, end OP and end barrier OP to MXNet engine. The function of each
        kind of OP is explained below.
        start barrier OP: barrier the start of a parent ByteTask, used to maintain original dependency.
        start OP: It notifies Core about task readiness. It is also used to delay the start of a child ByteTask.
        comm OP: the OP that does real communication, e.g., push, pull, allreduce.
        end OP: an OP that runs after a child ByteTask is finished. It notifies Core about the task completion.
        end barrier OP: an OP that runs after the parent ByteTask is finished, used to maintain original dependency.

        Below are several key data structures.

        self._tensor: a list of NDArrays of the same key of all devices. If push_pull, self._tensor includes push list
        and pull list of NDArrays of all devices.
        real: the original handle list of self._tensor, used for keep dependency.
        avatar: a new handle list of self._tensor.
        """

        if self.parent is None:
            if self.op == "push_pull":
                push_real = [t.handle for t in self._push_tensor] if isinstance(self._push_tensor, (tuple, list)) else [self._push_tensor.handle]
                pull_real = [t.handle for t in self._pull_tensor] if isinstance(self._pull_tensor, (tuple, list)) else [self._pull_tensor.handle]
                assert len(push_real) == len(pull_real)
                real = push_real + pull_real
            else:
                real = [t.handle for t in self._tensor] if isinstance(self._tensor, (tuple, list)) else [self._tensor.handle]
            avatar = []
            for h in real:
                avatar_h = NDArrayHandle()
                check_call(BYTESCHEDULER_LIB.bytescheduler_get_ndarray_avatar(
                    h, byref(avatar_h)))
                avatar.append(avatar_h)
            if self.op == "push_pull":
                # push avatar and pull avatar NDArrays
                self._avatar = [[NDArray(_) for _ in avatar[:int(len(avatar)/2)]], [NDArray(_) for _ in avatar[int(len(avatar)/2):]]]
                avatar = [_.handle for _ in self._avatar[0]] + [_.handle for _ in self._avatar[1]]
            else:
                self._avatar = [NDArray(_) for _ in avatar]
                avatar = [_.handle for _ in self._avatar]
        else:
            if self.op == "push_pull":
                push_real = [t.handle for t in self.parent._push_tensor] if isinstance(self.parent._push_tensor, (tuple, list)) else [self.parent._push_tensor.handle]
                pull_real = [t.handle for t in self.parent._pull_tensor] if isinstance(self.parent._pull_tensor, (tuple, list)) else [self.parent._pull_tensor.handle]
                real = push_real + pull_real
                push_avatar = []
                pull_avatar = []
                avatar = []
                for step_id in range(len(self._push_tensor)):
                    push_avatar.append([t.handle for t in self._push_tensor[step_id]] if isinstance(self._push_tensor[step_id], (tuple, list)) else [self._push_tensor[step_id].handle])
                    pull_avatar.append([t.handle for t in self._pull_tensor[step_id]] if isinstance(self._pull_tensor[step_id], (tuple, list)) else [self._pull_tensor[step_id].handle])
                    avatar += push_avatar[step_id] + pull_avatar[step_id]
            else:
                real = [t.handle for t in self.parent._tensor] if isinstance(self.parent._tensor, (tuple, list)) else [self.parent._tensor.handle]
                avatar = []
                for step_tensor in self._tensor:
                    avatar += [t.handle for t in step_tensor] if isinstance(step_tensor, (tuple, list)) else [step_tensor.handle]

        self._post_start_barrier(avatar, real)
        self._post_start_op(avatar)
        self._post_push_pull_barrier(avatar)

        # post real op
        if self.parent is None:
            self._post_communication(self._avatar)
        else:
            self._post_communication(self._tensor)

        self._post_end_op(avatar)

        self._post_end_barrier(avatar, real)

    # the push barrier is for barrier push-pull of all worker
    def _post_push_pull_barrier(self, avatar):
        if self.op == "push_pull":
            # push barrier and write dependency on barrier tensor and avatar with highest priority
            self._comm.push(self._barrier_key, self._barrier_tensor, 100000000-self.priority, is_barrier=True)
            deps = [self._barrier_tensor.handle] + avatar
            barrier_tensors_out = (NDArrayHandle * len(deps))(*deps)
            check_call(BYTESCHEDULER_LIB.bytescheduler_mxnet_barrier(
                barrier_tensors_out, 0,
                barrier_tensors_out, len(deps),
                100000000 - self.priority))

    def _post_start_barrier(self, avatar, real):
        """The start barrier is for keeping the original dependency. It does not need any callback."""
        if self.parent is None:
            barrier_tensors_in = (NDArrayHandle * len(real))(*real)
            if self.op == "push_pull":
                tensor_out = [self._barrier_tensor.handle]
            else:
                tensor_out = avatar
            barrier_tensors_out = (NDArrayHandle * len(tensor_out))(*tensor_out)
            check_call(BYTESCHEDULER_LIB.bytescheduler_mxnet_barrier(
                barrier_tensors_in, len(real),
                barrier_tensors_out, len(tensor_out),
                100000000 - self.priority))
        else:
            if hasattr(self.parent, "_posted_start_barrier"):
                return
            self.parent._posted_start_barrier = True
            if self.op == "push_pull":
                push_pull_barriers = []
                for child in self.parent.children:
                    push_pull_barriers.append(child._barrier_tensor.handle)
                deps = real + push_pull_barriers
            else:
                children_tensors = []
                for child in self.parent.children:
                    for chld_tensor in child._tensor:
                        if isinstance(chld_tensor, (tuple, list)):
                            for t in chld_tensor:
                                # including push tensor and pull tensor
                                if isinstance(t, (tuple, list)):
                                    children_tensors += [tt.handle for tt in t]
                                else:
                                    children_tensors += [t.handle]
                        else:
                            children_tensors += [chld_tensor.handle]
                deps = real + children_tensors
            barrier_tensors_out = (NDArrayHandle * len(deps))(*deps)
            check_call(BYTESCHEDULER_LIB.bytescheduler_mxnet_barrier(
                barrier_tensors_out, 0,
                barrier_tensors_out, len(deps),
                100000000 - self.priority))

    def _post_start_op(self, avatar):
        """The start op is only for notifying the Core about task ready. It does not add any dependency to the
        original DAG."""
        if self._immediate:
            return

        def start_callback(on_complete):
            if self._immediate:
                # call on_complete directly
                check_call(BYTESCHEDULER_LIB.bytescheduler_mxnet_on_complete(
                    c_void_p(on_complete)))
                return
            self._on_complete = on_complete
            self.notify_ready()

        # avoid garbage collection
        self._mxnet_start_callback = callback_t(start_callback)

        # post start op
        if self.op == "push_pull":
            tensor_out = (NDArrayHandle * 1)(*[self._barrier_tensor.handle])
            check_call(BYTESCHEDULER_LIB.bytescheduler_mxnet_op(
                tensor_out, 0, tensor_out, 1, self._mxnet_start_callback, 100000000-self.priority))
        else:
            tensor_out = (NDArrayHandle * len(avatar))(*avatar)
            check_call(BYTESCHEDULER_LIB.bytescheduler_mxnet_op(
                tensor_out, 0, tensor_out, len(avatar), self._mxnet_start_callback, 100000000-self.priority))

    def _post_end_op(self, avatar):
        """The end op is only for notifying the Core about task finishing. It does not add any dependency to the
        original DAG."""
        def end_callback(on_complete):
            # call on_complete directly
            check_call(BYTESCHEDULER_LIB.bytescheduler_mxnet_on_complete(
                c_void_p(on_complete)))
            self.notify_finish()

        # avoid garbage collection
        self._mxnet_end_callback = callback_t(end_callback)

        # post end op
        tensor_out = (NDArrayHandle * len(avatar))(*avatar)
        check_call(BYTESCHEDULER_LIB.bytescheduler_mxnet_op(
            tensor_out, 0, tensor_out, len(avatar), self._mxnet_end_callback, 100000000-self.priority))

    def _post_end_barrier(self, avatar, real):
        """The end barrier is for keeping the original dependency. It does not need any callback."""
        if self.parent is None:
            deps = real + avatar
            barrier_tensors_out = (NDArrayHandle * len(deps))(*deps)
            check_call(BYTESCHEDULER_LIB.bytescheduler_mxnet_barrier(
                barrier_tensors_out, 0,
                barrier_tensors_out, len(deps),
                100000000-self.priority))
        else:
            # _child_tensors is a list of avatar, and avatar itself is also a list
            if not hasattr(self.parent, "_children_tensors"):
                self.parent._children_tensors = [avatar]
            else:
                self.parent._children_tensors.append(avatar)
            if len(self.parent._children_tensors) == len(self.parent.children):
                tensors_in = [_ for sublist in self.parent._children_tensors for _ in sublist]
                barrier_tensors_in = (NDArrayHandle * len(tensors_in))(*tensors_in)
                barrier_tensors_out = (NDArrayHandle * len(real))(*real)
                check_call(BYTESCHEDULER_LIB.bytescheduler_mxnet_barrier(
                    barrier_tensors_in, len(tensors_in),
                    barrier_tensors_out, len(real),
                    100000000-self.priority))

    def _tensor_size(self):
        """Returns the size of one tensor of the task"""
        if self.op == "push_pull":
            if self.parent is not None:
                tensor = self._push_tensor[0]
            else:
                tensor = self._push_tensor
            assert isinstance(tensor, (tuple, list))
            return tensor[0].size
        else:
            if self.parent is not None:
                tensor = self._tensor[0]
            else:
                tensor = self._tensor
            if isinstance(tensor, (tuple, list)):
                return tensor[0].size
            else:
                return tensor.size

    def _partition_single_tensor(self, tensor, size, num_workers=None):
        """Only partition a single tensor.

        Arguments:
            size: An integer. After partitioning, each tensor partition size must be equal or smaller than `size`.

        Returns:
            A list of partitioned tensors.
        """
        number = (tensor.size - 1) // size + 1
        if num_workers:
            remainder = number % num_workers
            if remainder != 0:
                number = number + num_workers - remainder

        if number > tensor.shape[0]:
            self._logger.warning(
                "The number of tensor rows (with shape {}) is smaller than partition number {}.".format(tensor.shape, number))
            number = tensor.shape[0]
        num_per_partition = tensor.shape[0] // number
        partitions_with_extra = tensor.shape[0] % number

        partitions = []
        start = 0
        end = num_per_partition
        for i in range(number):
            handle = NDArrayHandle()
            check_call(BYTESCHEDULER_LIB.bytescheduler_get_ndarray_avatar(
                tensor.handle, byref(handle)))
            avatar = NDArray(handle)[start:end]
            partitions.append(avatar)
            start = end
            end += num_per_partition
            if i >= number - partitions_with_extra - 1:
                end += 1
        return partitions

    def _partition_tensor_list(self, tensors, size, num_workers=None):
        """Partition a list of tensors.

        Arguments:
            size: An integer. After partitioning, each tensor partition size must be equal or smaller than `size`.

        Returns:
            A list of partitioned tensors.
        """
        tot_partitions = []
        num_partitions = 0
        for tensor in tensors:
            partitions = self._partition_single_tensor(tensor, size, num_workers)
            if num_partitions:
                assert num_partitions == len(partitions)
            else:
                num_partitions = len(partitions)
            tot_partitions.append(partitions)

        # Group partitions with same index from each tensor
        ret_partitions = []
        for p in zip(*tot_partitions):
            ret_partitions.append(p)
        return ret_partitions

    def _partition_tensor(self, size, num_workers=None):
        """Zero-copy implementation.
        Note: ndarray works for up to ~4 billion parameters.

        Arguments:
            size: An integer. After partitioning, each tensor partition size must be equal or smaller than `size`.

        Returns:
            A list of partitioned tensors.
        """
        if self.op == "push_pull":
            assert isinstance(self._push_tensor, (tuple, list)) and isinstance(self._pull_tensor, (tuple, list))
            push_partitions = self._partition_tensor_list(self._push_tensor, size, num_workers)
            pull_partitions = self._partition_tensor_list(self._pull_tensor, size, num_workers)
            assert len(push_partitions) == len(pull_partitions)
            ret_partitions = []
            for p in zip(push_partitions, pull_partitions):
                ret_partitions.append(p)
            return ret_partitions
        else:
            if isinstance(self._tensor, (tuple, list)):
                return self._partition_tensor_list(self._tensor, size, num_workers)
            else:
                return self._partition_single_tensor(self._tensor, size, num_workers)

    def _immediate_do(self):
        self._prepare()

