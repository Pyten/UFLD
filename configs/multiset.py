# DATA
loader_workers = 8
seg_dataset = 'Bdd100k'
cls_dataset = 'Bdd100k'
seg_data_root = "/home/pantengteng/datasets/bdd100k/seg"
cls_data_root = "/home/pantengteng/datasets/bdd100k/"

# TRAIN
epoch = 100
batch_size = 4#32
# iters_per_ep =70000
optimizer = 'SGD' #'Adam'#['SGD','Adam']
learning_rate = 0.01 #0.1#0.001
weight_decay = 1e-3 #1e-4
momentum = 0.9

scheduler =  'cos' #['multi', 'cos']
steps = [25] #[25,38]
gamma  = 0.1
warmup = 'linear' #None#
warmup_iters = 200#100#695

# VAL
val = True
val_batch_size = 2

# NETWORK
use_cls = True
use_seg = True
# only_seg_road = False
# seg_class_num = 2 # optional
seg_class_num = 19
griding_num = 200
backbone = '101' #'18' #'101' # '18'

# LOSS
# awl = 2
cls_loss_w  = 1 #
seg_loss_w = 1 #2 #-1 #2
sim_loss_w = 0 #0.2#0.0# 0.1 relation_loss
shp_loss_w = 0 #0.1#0.0# 0.1 relation_dis

# EXP
note = ''

log_path = "/data/pantengteng/tensorboard_logs"# "/home/pantengteng/Programs/tensorboard_logs"

# FINETUNE or RESUME MODEL PATH
finetune = None #"./tusimple_18.pth"#None #/culane_18.pth"
resume = None
 
# TEST
test_model = "/data/pantengteng/tensorboard_logs/20201021_211207_lr_1e-02_b_2/ep099.pth"
#"/data/pantengteng/tensorboard_logs/20201015_110930_lr_1e-02_b_4/ep080.pth"#/20201014_105018_lr_1e-02_b_4/ep088.pth"#None#"../tensorboard_logs/20201011_170824_lr_1e-02_b_8/ep047.pth"#"../tensorboard_logs/20201010_184217_lr_1e-01_b_8/ep027.pth" #"./culane_18.pth"
test_work_dir = None
save_prefix = "new_2315_"

num_lanes = 4#14