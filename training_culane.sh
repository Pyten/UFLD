export CUDA_VISIBLE_DEVICES=3
export NGPUS=1
export OMP_NUM_THREADS=4 # you can change this value according to your number of cpu cores


python train.py configs/culane.py --batch_size 16
# python train.py configs/tusimple.py
