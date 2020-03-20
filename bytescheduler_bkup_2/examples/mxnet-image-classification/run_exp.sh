#!/bin/bash

# Utility functions
hecho () {
    echo -e "\e\033[0;33m[RUN EXP]\e[0m $1"
}

if [[ $# -ne 4 ]]; then
    hecho "Wrong number of arguments"
    hecho "Usage: $0 START END STEP SAVING_FOLDERNAME"
    exit 0
fi

PARTITION_START=$1
PARTITION_END=$2
PARTITION_STEP=$3
SAVING_FOLDERNAME=$4

hecho "Running exp on range: [$PARTITION_START, $PARTITION_END] with step $PARTITION_STEP"

mkdir -p "./$SAVING_FOLDERNAME"

hecho "Saving results in $SAVING_FOLDERNAME"

for ((PARTITION=$PARTITION_START; PARTITION<=$PARTITION_END; PARTITION+=$PARTITION_STEP)); do
    hecho "Running script with parameter $PARTITION"
    ./run.sh $PARTITION 2>&1 | tee ./${SAVING_FOLDERNAME}/${PARTITION}_output.txt
    sleep 1s
    pkill python
    hecho "Killed all python processes."
    sleep 2s
done

hecho "Finished"

