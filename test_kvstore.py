import bytescheduler as bs
import mxnet as mx

from bytescheduler.mxnet.kvstore import ScheduledKVStore

kv = mx.kvstore.create("dist_sync")
kv = ScheduledKVStore(kv)

rank = kv.rank
print("[{}] Started.".format(rank))

shape = (4,3)

print("[{}] Testing init.".format(rank))
kv.init('3', mx.nd.ones(shape)*2)
a = mx.nd.zeros(shape)
kv.pull('3', out=[a])
print ("[{}] inited out:".format(rank), a.asnumpy())

print("[{}] Testing push pull.".format(rank))
shape = (4,3)
kv.init("large_tensor", mx.nd.zeros(shape))
pushd = mx.nd.random.normal(shape=shape)
print("[{}] Pushing ".format(rank), pushd.asnumpy())
kv.push("large_tensor", [pushd])
b = mx.nd.zeros(shape)
kv.pull("large_tensor", out=[b])
print("[{}] Got ".format(rank), b.asnumpy())

for i in range(4):
    print("[{}] Testing push pull without init.".format(rank))
    shape = (4,3)
    pushd = mx.nd.random.normal(shape=shape)
    print("[{}] Pushing ".format(rank), pushd.asnumpy())
    kv.push("large_tensor", [pushd])
    b = mx.nd.zeros(shape)
    kv.pull("large_tensor", out=[b])
    print("[{}] Got ".format(rank), b.asnumpy())


