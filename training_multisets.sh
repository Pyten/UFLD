export CUDA_VISIBLE_DEVICES=3
export NGPUS=1
export OMP_NUM_THREADS=4 # you can change this value according to your number of cpu cores 


python train_multi_datasets.py configs/multiset.py --batch_size 2 --epoch 50
# python train.py configs/tusimple.py   --local_rank 0
