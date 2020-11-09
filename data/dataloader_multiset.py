import torch, os
import numpy as np

import torchvision.transforms as transforms
import data.mytransforms as mytransforms
from data.constant import tusimple_row_anchor, culane_row_anchor
# Pyten-20201022-ChangeDataset
from data.dataset_seg_cls import SegDataset, BddLaneClsDataset
from data.dataset import LaneClsDataset


def get_cls_train_loader(batch_size, data_root, griding_num, dataset, distributed, num_lanes, cfg, use_seg=False):
    target_transform = transforms.Compose([
        mytransforms.FreeScaleMask((cfg.size_h, cfg.size_w)),
        mytransforms.MaskToTensor(),
    ])
    segment_transform = transforms.Compose([
        mytransforms.FreeScaleMask((cfg.size_h, cfg.size_w)),
        mytransforms.MaskToTensor(),
    ])
    img_transform = transforms.Compose([
        transforms.Resize((cfg.size_h, cfg.size_w)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    simu_transform = mytransforms.Compose2([
        mytransforms.RandomRotate(6),
        mytransforms.RandomUDoffsetLABEL(100),
        mytransforms.RandomLROffsetLABEL(200)
    ])
    # Pyten-20201010-AddBddTrans
    segment_transform_bdd = transforms.Compose([
        mytransforms.FreeScaleMask((cfg.size_h, cfg.size_w)),
        mytransforms.MaskToTensor(),
    ])
    
    if dataset == 'CULane':
        # Pyten-20201023-Add anchorsParam
        cfg.anchors = culane_row_anchor
        train_dataset = LaneClsDataset(data_root,
                                           os.path.join(data_root, 'list/train_gt.txt'),
                                           img_transform=img_transform, target_transform=target_transform,
                                           simu_transform = simu_transform,
                                           segment_transform=segment_transform,
                                           row_anchor = cfg.anchors,
                                           griding_num=griding_num, use_seg=use_seg, num_lanes = num_lanes)
        cls_num_per_lane = 18
    # Pyten-20201010-AddBdd
    elif dataset == 'Bdd100k':
        # Pyten-20201023-Add anchorsParam
        cfg.anchors = tusimple_row_anchor
        # Pyten-20201021-OnlySegRoadandOthers
        train_dataset = BddLaneClsDataset(data_root,
                                           os.path.join(data_root, 'train.txt'), #new_train.txt
                                           img_transform=img_transform, target_transform=target_transform,
                                           simu_transform = simu_transform,
                                           griding_num=griding_num, 
                                           row_anchor = cfg.anchors,
                                           segment_transform=segment_transform_bdd,
                                           num_lanes = num_lanes, mode="/100k/train")
        # cls_num_per_lane = 56
        cls_num_per_lane = 56

    elif dataset == 'Tusimple':
        # Pyten-20201023-Add anchorsParam
        cfg.anchors = tusimple_row_anchor
        train_dataset = LaneClsDataset(data_root,
                                           os.path.join(data_root, 'train_gt.txt'),
                                           img_transform=img_transform, target_transform=target_transform,
                                           simu_transform = simu_transform,
                                           griding_num= griding_num, 
                                           row_anchor = cfg.anchors,
                                           segment_transform=segment_transform,use_seg=use_seg, num_lanes = num_lanes)
        cls_num_per_lane = 56
    
    else:
        raise NotImplementedError

    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        sampler = torch.utils.data.RandomSampler(train_dataset)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler = sampler, num_workers=cfg.loader_workers)

    return train_loader, cls_num_per_lane

def get_seg_train_loader(batch_size, data_root, dataset, distributed, cfg):
    target_transform = transforms.Compose([
        mytransforms.FreeScaleMask((cfg.size_h, cfg.size_w)),
        mytransforms.MaskToTensor(),
    ])
    img_transform = transforms.Compose([
        transforms.Resize((cfg.size_h, cfg.size_w)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    simu_transform = mytransforms.Compose2([
        mytransforms.RandomRotate(6),
        mytransforms.RandomUDoffsetLABEL(100),
        mytransforms.RandomLROffsetLABEL(200)
    ])
    
    if dataset == 'Bdd100k':
        train_dataset = SegDataset(data_root,
                                           os.path.join(data_root, 'train_seg.txt'), 
                                           img_transform=img_transform, target_transform=target_transform,
                                           simu_transform = simu_transform,
                                           mode="train")

    elif dataset == 'CityScape':
        train_dataset = SegDataset(data_root,
                                           os.path.join(data_root, 'train_gt.txt'),
                                           img_transform=img_transform, target_transform=target_transform,
                                           simu_transform = simu_transform,
                                           mode="train")

    else:
        raise NotImplementedError

    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        sampler = torch.utils.data.RandomSampler(train_dataset)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler = sampler, num_workers=cfg.loader_workers) #4

    return train_loader

# Pyten-20201019-Add_val_loader
def get_cls_val_loader(batch_size, data_root, griding_num, dataset, distributed, num_lanes, cfg, use_seg=False):
    target_transform = transforms.Compose([
        mytransforms.FreeScaleMask((cfg.size_h, cfg.size_w)),
        mytransforms.MaskToTensor(),
    ])
    img_transform = transforms.Compose([
        transforms.Resize((cfg.size_h, cfg.size_w)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    segment_transform = transforms.Compose([
        mytransforms.FreeScaleMask((cfg.size_h, cfg.size_w)),
        mytransforms.MaskToTensor(),
    ])
    # Pyten-20201010-AddBddTrans
    segment_transform_bdd = transforms.Compose([
        mytransforms.FreeScaleMask((cfg.size_h, cfg.size_w)),
        mytransforms.MaskToTensor(),
    ])
    
    if dataset == 'CULane':
        val_dataset = LaneClsDataset(data_root,
                                           os.path.join(data_root, 'list/val_gt.txt'),
                                           img_transform=img_transform, target_transform=target_transform,
                                           simu_transform = None,
                                           segment_transform=segment_transform,
                                           row_anchor = cfg.anchors,
                                           griding_num=griding_num, use_seg=use_seg, num_lanes = num_lanes)

    elif dataset == 'Bdd100k':
        val_dataset = BddLaneClsDataset(data_root,
                                           os.path.join(data_root, 'val.txt'),
                                           img_transform=img_transform, target_transform=target_transform,
                                           simu_transform = None,
                                           griding_num=griding_num,
                                           row_anchor = cfg.anchors,
                                           segment_transform=segment_transform_bdd,
                                           num_lanes = num_lanes, mode="/100k/val")

    elif dataset == 'Tusimple':
        val_dataset = LaneClsDataset(data_root,
                                           os.path.join(data_root, 'train_gt.txt'),
                                           img_transform=img_transform, target_transform=target_transform,
                                           simu_transform = None,
                                           griding_num=griding_num,
                                           row_anchor = cfg.anchors,
                                           segment_transform=segment_transform,use_seg=use_seg, num_lanes = num_lanes)
    else:
        raise NotImplementedError

    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    else:
        sampler = torch.utils.data.RandomSampler(val_dataset)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, sampler = sampler, num_workers=cfg.loader_workers) #4

    return val_loader

def get_seg_val_loader(batch_size, seg_data_root, dataset, distributed, cfg):
    target_transform = transforms.Compose([
        mytransforms.FreeScaleMask((cfg.size_h, cfg.size_w)),
        mytransforms.MaskToTensor(),
    ])
    img_transform = transforms.Compose([
        transforms.Resize((cfg.size_h, cfg.size_w)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    if dataset == 'Bdd100k':
        val_dataset = SegDataset(seg_data_root,
                                           os.path.join(seg_data_root, 'val_seg.txt'), 
                                           img_transform=img_transform, target_transform=target_transform,
                                           simu_transform = None,
                                           mode="val")

    elif dataset == 'CityScape':
        val_dataset = SegDataset(data_root,
                                           os.path.join(data_root, 'train_gt.txt'),
                                           img_transform=img_transform, target_transform=target_transform,
                                           simu_transform = None,
                                           mode="val")

    else:
        raise NotImplementedError

    if distributed:
        # batch_size = batch_size / args.n
        sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    else:
        sampler = torch.utils.data.RandomSampler(val_dataset)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, sampler = sampler, num_workers=cfg.loader_workers) #4

    return val_loader
