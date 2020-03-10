./local_mpi.sh $1 2 2 python train_imagenet.py --network resnet --num-layers 18 --gpus 0,1 --benchmark 1 --kv-store dist_sync --batch-size 96 --disp-batches 10 --num-examples 8000 --num-epochs 1 
#--profile-server-suffix server_profile.json
