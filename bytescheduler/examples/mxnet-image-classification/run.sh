if [ $# -lt 1 ]; then
    echo "usage: $0 partition"
    exit -1;
fi

./local.sh $1 4 4 python train_imagenet.py --network resnet --num-layers 18 --gpus 0,1,2,3 --benchmark 1 --kv-store dist_sync --batch-size 96 --disp-batches 10 --num-examples 8000 --num-epochs 1 
#--profile-server-suffix server_profile.json
