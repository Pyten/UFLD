import os, argparse
from utils.dist_utils import is_main_process, dist_print, DistSummaryWriter
from utils.config import Config
import torch
# Pyten
import pdb
from data.cityscapes_labels import trainId2color

# Pyten-202010-DecodePredict
def decode_seg_color_map(label):
    # pdb.set_trace()
    color_map = torch.ones((label.shape[0], label.shape[1], 3))* 255
    for id in trainId2color:
        color_map[label == id] = torch.tensor(trainId2color[id]).float()

    return color_map

def decode_cls_color_map(img, label, cfg): # img, 
    """
    img: transformed img，tensor
    label: prediction or label
    cfg: anchors,griding_num
    """
    # ?(288,800,3)
    c, h, w  = img.shape
    grid_width = w / cfg.griding_num
    anchor_height = cfg.anchors[1] - cfg.anchors[0]
    # if h != 288:
    #         scale_f = lambda x : int((x * 1.0/288) * h)
    #         anchor_list = list(map(scale_f, anchors))
    
    # Pyten-20201012-fixshape
    # Pyten-20201013-AddExtraCell
    # color_map = torch.ones((img.shape[1], img.shape[2], img.shape[0]))* 255
    color_map = torch.ones((img.shape[1], img.shape[2] + int(grid_width+1), img.shape[0]))* 255
    for anchor in range(label.shape[0]):
        for lane in range(label.shape[1]):
            grid_index = label[anchor, lane]
            # pdb.set_trace()
            # if grid_index < cfg.griding_num:
            color_map[cfg.anchors[anchor] - anchor_height:cfg.anchors[anchor], int(grid_width*(grid_index)):int(grid_width*(grid_index + 1)), :] =  torch.tensor(trainId2color[lane]).float()
    return color_map


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help = 'path to config file')
    parser.add_argument('--local_rank', type=int, default=0)

    parser.add_argument('--dataset', default = None, type = str)
    parser.add_argument('--data_root', default = None, type = str)
    parser.add_argument('--epoch', default = None, type = int)
    parser.add_argument('--batch_size', default = None, type = int)
    parser.add_argument('--optimizer', default = None, type = str)
    parser.add_argument('--learning_rate', default = None, type = float)
    parser.add_argument('--weight_decay', default = None, type = float)
    parser.add_argument('--momentum', default = None, type = float)
    parser.add_argument('--scheduler', default = None, type = str)
    parser.add_argument('--steps', default = None, type = int, nargs='+')
    parser.add_argument('--gamma', default = None, type = float)
    parser.add_argument('--warmup', default = None, type = str)
    parser.add_argument('--warmup_iters', default = None, type = int)
    parser.add_argument('--backbone', default = None, type = str)
    parser.add_argument('--griding_num', default = None, type = int)
    parser.add_argument('--use_seg', default = None, type = str2bool)
    # Pyten-20201020-SeperateModels
    parser.add_argument('--use_cls', default = True, type = str2bool)
    parser.add_argument('--only_seg_road', default = False, type = str2bool)
    # Pyten-20201021-AddSegClassNumParam
    parser.add_argument('--seg_class_num', default=19, type=int)

    parser.add_argument('--sim_loss_w', default = None, type = float)
    parser.add_argument('--shp_loss_w', default = None, type = float)
    parser.add_argument('--note', default = None, type = str)
    parser.add_argument('--log_path', default = None, type = str)
    parser.add_argument('--finetune', default = None, type = str)
    parser.add_argument('--resume', default = None, type = str)
    parser.add_argument('--test_model', default = None, type = str)
    parser.add_argument('--test_work_dir', default = None, type = str)
    parser.add_argument('--num_lanes', default = None, type = int)
    return parser

def merge_config():
    args = get_args().parse_args()
    cfg = Config.fromfile(args.config)

    items = ['dataset','data_root','epoch','batch_size','optimizer','learning_rate',
    'weight_decay','momentum','scheduler','steps','gamma','warmup','warmup_iters',
    'use_seg','use_cls','only_seg_road','griding_num','backbone','sim_loss_w','shp_loss_w','note','log_path',
    'finetune','resume', 'test_model','test_work_dir', 'num_lanes', 'seg_class_num']
    for item in items:
        if getattr(args, item) is not None:
            dist_print('merge ', item, ' config')
            setattr(cfg, item, getattr(args, item))

    # Pyten-20201020-SeperateModels
    assert(cfg.use_cls or cfg.use_seg)

    if cfg.only_seg_road:
        cfg.seg_class_num = 2
    # pdb.set_trace()
    if "val_batch_size" not in cfg:
        cfg.val_batch_size = cfg.batch_size

    return args, cfg


def save_model(net, optimizer, epoch,save_path, distributed):
    if is_main_process():
        model_state_dict = net.state_dict()
        state = {'model': model_state_dict, 'optimizer': optimizer.state_dict()}
        # state = {'model': model_state_dict}
        assert os.path.exists(save_path)
        model_path = os.path.join(save_path, 'ep%03d.pth' % epoch)
        torch.save(state, model_path)

import pathspec

def cp_projects(to_path):
    if is_main_process():
        with open('./.gitignore','r') as fp:
            ign = fp.read()
        ign += '\n.git'
        spec = pathspec.PathSpec.from_lines(pathspec.patterns.GitWildMatchPattern, ign.splitlines())
        all_files = {os.path.join(root,name) for root,dirs,files in os.walk('./') for name in files}
        matches = spec.match_files(all_files)
        matches = set(matches)
        to_cp_files = all_files - matches
        # to_cp_files = [f[2:] for f in to_cp_files]
        # pdb.set_trace()
        for f in to_cp_files:
            dirs = os.path.join(to_path,'code',os.path.split(f[2:])[0])
            if not os.path.exists(dirs):
                os.makedirs(dirs)
            os.system('cp %s %s'%(f,os.path.join(to_path,'code',f[2:])))


import datetime, os
def get_work_dir(cfg):
    now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    hyper_param_str = '_lr_%1.0e_b_%d' % (cfg.learning_rate, cfg.batch_size)
    work_dir = os.path.join(cfg.log_path, now + hyper_param_str + cfg.note)
    return work_dir

def get_logger(work_dir, cfg):
    logger = DistSummaryWriter(work_dir)
    config_txt = os.path.join(work_dir, 'cfg.txt')
    if is_main_process():
        with open(config_txt, 'w') as fp:
            fp.write(str(cfg))

    return logger
     