export CUDA_VISIBLE_DEVICES=1
export NGPUS=1
export OMP_NUM_THREADS=4 # you can change this value according to your number of cpu cores 


python train.py configs/bdd100k2.py --batch_size 2
# python train.py configs/tusimple.py
