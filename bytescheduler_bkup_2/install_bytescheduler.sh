export USE_BYTESCHEDULER=1
export BYTESCHEDULER_WITH_MXNET=1
export BYTESCHEDULER_WITHOUT_PYTORCH=1
export MXNET_ROOT=/research/d2/jc1901/proposed-scheduler/incubator-mxnet-1.5.x
#export MXNET_ROOT=/research/d2/jc1901/incubator-mxnet-1.5.x-priority

python setup.py install
