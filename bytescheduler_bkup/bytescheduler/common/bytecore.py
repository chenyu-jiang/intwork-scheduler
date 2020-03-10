#!/usr/bin/python

from __future__ import absolute_import
import sys

try:
    import queue
except ImportError:
    import Queue as queue
import threading
import logging
import collections
from .bytetask import ByteTask
from .profiler import Profiler

import bytescheduler.proposed as proposed

import os

def get_random_server(key):
    return (key * 9973) % proposed.get_world_size()


class ByteCore(object):
    """The core of ByteScheduler. Once Core gets a ByteTask (which represents a communication operation, e.g., push,
    allreduce), it partitions the ByteTask and decides when to send each partition according to priority."""

    def __init__(self, logger=None):
        """
        Args:
            logger: ByteScheduler logger object
        """
        if logger is None:
            self._logger = logging.getLogger("ByteScheduler")
        else:
            self._logger = logger

        # A priority queue of ByteTask, tasks are sorted according to its priority.
        self._queue = queue.PriorityQueue()

        self._is_started = False

        # DATA represents normal tasks and EXIT signals the scheduler thread to be terminated.
        self._commands = {'DATA': 0, 'EXIT': 1}

        # Control credit
        self._running_lock = threading.Lock()

        # Pending tasks that are not ready
        self._pending = set()
        self._pending_lock = threading.Lock()

        # Only used to avoid task being garbage collected before completion.
        self._running = set()
        self._finished = collections.OrderedDict()

        # The rank of a worker
        self._rank = None

        # The communication architecture used, e.g., ps or allreduce.
        self._arch = None

        # Partition unit, i.e., the number of parameters
        self._partition = int(os.environ.get('BYTESCHEDULER_PARTITION', 100000))

        # Credit, i.e., the max number of unacknowledged parameters
        self._credit = float(os.environ.get('BYTESCHEDULER_CREDIT', 4000000))
        self._credit_limit = self._credit

        # We expect that the first key is same across iterations and we use it to count how many training steps have
        # been run.
        self._first_key = None
        self._step = 0

        # profiling
        self._timeline = os.environ.get('BYTESCHEDULER_TIMELINE', '')
        self._profiler = None

    def start(self, arch):
        """Start core.
        Args:
            rank: the rank of the worker
            arch: the communication architecture, "ps" or "allreduce"
        """
        if self._is_started:
            self._logger.warning("Core is already started.")
            return

        # Setup profiler
        if self._rank == 0 and self._timeline:
            self._logger.info("rank {}: profiler is enabled.".format(self._rank))
            self._profiler = Profiler(self._timeline)
        else:
            self._profiler = Profiler('')

        assert arch == "ps" or arch == "allreduce", arch + " not supported!"
        self._arch = arch


        # Initialize proposed scheduler
        proposed.init()

        self._rank = proposed.get_rank()

        self._is_started = True

        self._logger.info(
            "start Core {}: credit {}, partition {}.".format(
                self._rank, self._credit, self._partition))

    def shutdown(self, wait_for_all=False):
        """Shut Core down.

        Args:
            wait_for_all: Flag indicating whether to wait completion of undone tasks.
        """
        if not self._is_started:
            self._logger.warning("Core is already shutdown.")
            return
        if wait_for_all:
            self._queue.put((sys.maxint, self._commands['EXIT'], None))
        else:
            self._queue.put((-sys.maxint, self._commands['EXIT'], None))
        with self._running_lock:
            self._credit = sys.maxint
        self._is_started = False
        self._profiler.stop()
        self._logger.info("shutdown Core {}.".format(self._rank))

    def post(self, task):
        """Post a communication task to Core for scheduling.
        Args:
            task: a ByteTask object
        Returns:
            A boolean value indicating whether the task is successfully posted
        """
        if not self._is_started:
            self._logger.error("Core is not running, call start first!")
            return False

        if not isinstance(task, ByteTask):
            self._logger.error(
                "{} is not an instance of ByteTask!".format(task.desc))
            return False
        else:
            # Set the first key and use it to count number of training steps.
            if not self._first_key:
                self._first_key = task.name
            if self._first_key == task.name:
                self._step += 1

            # Partition a task if its tensor is larger than a threshold.
            if task.tensor_size() > self._partition:
                subtasks = task.partition(size=self._partition)
            else:
                task.set_assigned_server(get_random_server(task.id))
                subtasks = [task]
            
            # print("Tensor {} have {} partitions.".format(task.name, len(subtasks)))

            # A task will bypass scheduling and start immediately after partition if immediate is True.
            if task.is_immediate():
                # The callback runs after an immediate task is finished.
                def _end_callback(t, self):
                    with self._running_lock:
                        if t not in self._running:
                            raise RuntimeError("{} not in _running".format(t.desc))
                        self._running.remove(t)
                        self._finished[t.name] = t
                    # print("[{}] Immediate task {} with op {} finished.".format(proposed.get_rank(), task.name, task.op))
                    self._profiler.put(t.name, t.op + 'COMMUNICATION', 'E')

                for t in subtasks:
                    with self._running_lock:
                        self._running.add(t)
                    self._profiler.put(t.name, t.op + 'COMMUNICATION', 'B')
                    t.immediate_do(callback=_end_callback, callback_context=self)
                return True

            # The callback runs when a non-immediate task is ready.
            def _start_callback(task, self):
                with self._pending_lock:
                    self._pending.remove(task)
                with self._running_lock:
                    self._running.add(task)

            # The callback runs after an non-immediate task is finished.
            def _end_callback(task, self):
                # print("End callback called with tensor {}, id {}.".format(task.name, task.id))
                with self._running_lock:
                    if task not in self._running:
                        raise RuntimeError("{} not in _running".format(task.desc))
                    self._running.remove(task)
                    self._finished[task.name] = task

            # Prepare the task, i.e., add dependency Proxies.
            for t in subtasks:
                with self._pending_lock:
                    self._pending.add(t)
                t.register_end_callback(callback=_end_callback, callback_context=self)
                t.prepare(start_callback=_start_callback, start_callback_context=self)
            return True

# Init a core once the module is imported
core = ByteCore()
