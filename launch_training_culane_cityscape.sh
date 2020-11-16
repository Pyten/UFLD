export CUDA_VISIBLE_DEVICES=0,1
export NGPUS=2 #4
export OMP_NUM_THREADS=8 # you can change this value according to your number of cpu cores


python -m torch.distributed.launch --nproc_per_node=$NGPUS --master_port 29731 train_multi_datasets.py configs/multiset_culane_cityscape.py --distributed True --learning_rate 0.1 --epoch 30 --resume "/data/pantengteng/tensorboard_logs/20201111_102646_lr_1e-01_b_4/ep009.pth"
# python train.py configs/tusimple.py
# train_multi_datasets.py configs/multiset.py 
# train.py configs/bdd100k.py
