#!/bin/bash
# set -x
if [ $# -lt 3 ]; then
    echo "usage: $0 num_servers num_workers bin [args..]"
    exit -1;
fi

export USE_BYTESCHEDULER=1
export BYTESCHEDULER_CREDIT_TUNING=0

export RANK=$1
echo "RANK= ${RANK}"
shift

export BYTESCHEDULER_PARTITION=$1
echo "PARTITION= ${BYTESCHEDULER_PARTITION}"
shift
# export BYTESCHEDULER_TUNING=1
# export BYTESCHEDULER_PARTITION=512000
#export BYTESCHEDULER_CREDIT=10240000
# export BYTESCHEDULER_TIMELINE=timeline.json
# export BYTESCHEDULER_DEBUG=1

export DMLC_NUM_SERVER=$1
echo "NUM_SERVER = ${DMLC_NUM_SERVER}"
shift
export DMLC_NUM_WORKER=$1
echo "NUM_WORKER = ${DMLC_NUM_WORKER}"
shift
bin=$1
shift
arg="$@"

# start the scheduler
export DMLC_PS_ROOT_URI='137.189.88.83'
export DMLC_PS_ROOT_PORT=8000

if [ ${RANK} -eq 0 ]
then
    echo "Starting scheduler on rank 0."
    export DMLC_ROLE='scheduler'
    ${bin} ${arg} &
fi


# start server
export DMLC_ROLE='server'
export HEAPPROFILE=./S${RANK}
${bin} ${arg} &

# start worker
export DMLC_ROLE='worker'
export HEAPPROFILE=./W${RANK}
${bin} ${arg} &

wait
