from .proposed import ProposedWrapper as _ProposedWrapper
import os

_suffix = "build/libproposed.dylib"

_wrapper = _ProposedWrapper(os.path.join(os.path.dirname(__file__), _suffix))

init = _wrapper.init
get_rank = _wrapper.get_rank
get_world_size = _wrapper.get_world_size
post_tensor = _wrapper.post_tensor
proposed_signal_partition_finished = _wrapper.proposed_signal_partition_finished
