# DATA
dataset='CULane'
data_root = "/home/pantengteng/datasets/CULane"

# TRAIN
epoch = 50
batch_size = 32
optimizer = 'SGD'  #['SGD','Adam']
learning_rate = 0.1
weight_decay = 1e-4
momentum = 0.9

scheduler = 'multi' #['multi', 'cos']
steps = [25,38]
gamma  = 0.1
warmup = 'linear'
warmup_iters = 695

# NETWORK
use_seg = True
griding_num = 200
backbone = '18'

# LOSS
sim_loss_w = 0.0
shp_loss_w = 0.0

# EXP
note = ''

log_path = "/home/pantengteng/Programs/tensorboard_logs"

# FINETUNE or RESUME MODEL PATH
finetune = None
resume = None

# TEST
test_model = "../tensorboard_logs/20200909_134451_lr_1e-01_b_8/ep020.pth"#"./culane_18.pth"# 
test_work_dir = None

num_lanes = 4




