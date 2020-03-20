if [ $# -lt 2 ]; then
    echo "usage: $0 rank partition"
    exit -1;
fi

./distributed.sh $1 $2 2 2 python train_imagenet.py --network vgg --num-layers 16 --gpus 0 --benchmark 1 --kv-store dist_sync --batch-size 96 --disp-batches 10 --num-examples 8000 --num-epochs 1 
#--profile-server-suffix server_profile.json
