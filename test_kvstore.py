import bytescheduler as bs
import mxnet as mx

from bytescheduler.mxnet.kvstore import ScheduledKVStore
import bytescheduler.proposed as proposed

kv = mx.kvstore.create("dist_sync")
kv = ScheduledKVStore(kv)

rank = proposed.get_rank()
print("[{}] Started.".format(rank))

shape = (2,3)

print("[{}] Testing init.".format(rank))
kv.init('3', mx.nd.ones(shape)*2)
a = mx.nd.zeros(shape)
kv.pull('3', out=[a])
print ("[{}] inited out:".format(rank), a.asnumpy())

print("[{}] Testing push pull.".format(rank))
shape = (100000,3)
kv.init("large_tensor", mx.nd.zeros(shape))
kv.push("large_tensor", [mx.nd.ones(shape)*2])
b = mx.nd.zeros(shape)
kv.pull("large_tensor", out=[b])
print("[{}] Got ".format(rank), b.asnumpy())

