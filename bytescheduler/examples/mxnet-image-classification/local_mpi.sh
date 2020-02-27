#!/bin/bash
# set -x
if [ $# -lt 3 ]; then
    echo "usage: $0 num_servers num_workers bin [args..]"
    exit -1;
fi

export USE_BYTESCHEDULER=1
# export BYTESCHEDULER_TUNING=1
# export BYTESCHEDULER_PARTITION=512000
# export BYTESCHEDULER_CREDIT=4096000
# export BYTESCHEDULER_TIMELINE=timeline.json
# export BYTESCHEDULER_DEBUG=1

export DMLC_NUM_SERVER=$1
shift
export DMLC_NUM_WORKER=$1
shift
bin=$1
shift
arg="$@"

# start the scheduler
export DMLC_PS_ROOT_URI='127.0.0.1'
export DMLC_PS_ROOT_PORT=8000
export DMLC_ROLE='scheduler'
${bin} ${arg} &


# start servers
export DMLC_ROLE='server'
for ((i=0; i<${DMLC_NUM_SERVER}; ++i)); do
    export HEAPPROFILE=./S${i}
    ${bin} ${arg} &
done

# Here MPI kicks in
# mpi command to start workers
export DMLC_ROLE='worker'

MPI_CMD="mpirun "
for ((i=0; i<${DMLC_NUM_WORKER}; ++i)); do
    MPI_CMD="${MPI_CMD} -np 1 -x HEAPPROFILE=./W${i} ${bin} ${arg}"
    if [ ${i} -ne $((${DMLC_NUM_WORKER} -1)) ]
    then 
        MPI_CMD="${MPI_CMD} :"
    fi
done


echo -e "\e\033[0;33m[EXECUTING]\e[0m ${MPI_CMD}"
eval ${MPI_CMD}
