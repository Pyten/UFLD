import torch, os, datetime
import numpy as np

import pdb
from model.model import parsingNet
from data.dataloader import get_train_loader, get_val_loader

from utils.dist_utils import dist_print, dist_tqdm, is_main_process, DistSummaryWriter
from utils.factory import get_metric_dict, get_loss_dict, get_optimizer, get_scheduler
from utils.metrics import MultiLabelAcc, AccTopk, Metric_mIoU, update_metrics, reset_metrics

from utils.common import merge_config, save_model, cp_projects, decode_seg_color_map, decode_cls_color_map
from utils.common import get_work_dir, get_logger

from utils.AutomaticWeightedLoss import AutomaticWeightedLoss

import time

from data.constant import tusimple_row_anchor, culane_row_anchor

def inference(net, data_label, use_seg, use_cls):
    if use_seg and use_cls:
        img, cls_label, seg_label = data_label
        img, cls_label, seg_label = img.cuda(), cls_label.cuda(), seg_label.cuda()
        cls_out, seg_out = net(img)
        # Pyten-20201010-ChangeSegOut
        # seg_out = torch.max(seg_out, dim=1)[1].float()
        return {'cls_out': cls_out, 'cls_label': cls_label, 'seg_out':seg_out, 'seg_label': seg_label}
    elif use_seg:
        img, cls_label, seg_label = data_label
        img, seg_label = img.cuda(), seg_label.cuda()
        seg_out = net(img)
        return {'seg_out':seg_out, 'seg_label': seg_label}
    else:
        img, cls_label = data_label
        img, cls_label = img.cuda(), cls_label.cuda()
        cls_out = net(img)
        return {'cls_out': cls_out, 'cls_label': cls_label}


def resolve_val_data(results, use_seg, use_cls):
    if use_cls:
        results['cls_out'] = torch.argmax(results['cls_out'], dim=1)
    if use_seg:
        results['seg_out'] = torch.argmax(results['seg_out'], dim=1)
    return results


def calc_loss(loss_dict, results, logger, global_step, mode = "train", awl=None):
    loss = 0
    # Pyten-20201015-AddAutoWeightedLoss
    loss_list = []
    for i in range(len(loss_dict['name'])):

        data_src = loss_dict['data_src'][i]

        datas = [results[src] for src in data_src]

        loss_cur = loss_dict['op'][i](*datas)

        if global_step % 20 == 0:
            logger.add_scalar(mode + '/loss/'+loss_dict['name'][i], loss_cur, global_step)

        if loss_dict['weight'][i] != -1:
            loss += loss_cur * loss_dict['weight'][i]
        else:
            loss_list.append(loss_cur)

    if awl:
        loss = awl(loss, *loss_list)
    
    return loss


def train(net, data_loader, loss_dict, optimizer, scheduler,logger, epoch, metric_dict, use_seg, use_cls, awl):
    net.train()
    progress_bar = dist_tqdm(data_loader)
    t_data_0 = time.time()
    # Pyten-20201019-FixBug
    reset_metrics(metric_dict)
    total_loss = 0
    for b_idx, data_label in enumerate(progress_bar):
        t_data_1 = time.time()
        # reset_metrics(metric_dict)
        global_step = epoch * len(data_loader) + b_idx

        t_net_0 = time.time()
        # pdb.set_trace()
        results = inference(net, data_label, use_seg, use_cls)
        loss = calc_loss(loss_dict, results, logger, global_step, "train",awl)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(global_step)
        t_net_1 = time.time()

        results = resolve_val_data(results, use_seg, use_cls)

        update_metrics(metric_dict, results)
        if global_step % 20 == 0:
            # Pyten-20201012-AddImage2TBD
            logger.add_image("train_image", data_label[0][0], global_step=global_step)
            if use_seg:
                seg_color_out = decode_seg_color_map(results["seg_out"][0])
                seg_color_label = decode_seg_color_map( results["seg_label"][0])
                logger.add_image("train_seg/predict", seg_color_out, global_step=global_step, dataformats='HWC')
                logger.add_image("train_seg/label",seg_color_label, global_step=global_step, dataformats='HWC')
            # Pyten-20201012-anchors需要根据参数调整 
            if use_cls:
                cls_color_out = decode_cls_color_map(data_label[0][0], results["cls_out"][0], cfg)
                cls_color_label = decode_cls_color_map(data_label[0][0], results["cls_label"][0], cfg)
                logger.add_image("train_cls/predict", cls_color_out, global_step=global_step, dataformats='HWC')
                logger.add_image("train_cls/label", cls_color_label, global_step=global_step, dataformats='HWC')
            #  results: {'cls_out': cls_out, 'cls_label': cls_label, 'seg_out':seg_out, 'seg_label': seg_label}

            for me_name, me_op in zip(metric_dict['name'], metric_dict['op']):
                logger.add_scalar('train_metric/' + me_name, me_op.get(), global_step=global_step)
        logger.add_scalar('train/meta/lr', optimizer.param_groups[0]['lr'], global_step=global_step)

        if hasattr(progress_bar,'set_postfix'):
            kwargs = {me_name: '%.3f' % me_op.get() for me_name, me_op in zip(metric_dict['name'], metric_dict['op'])}
            progress_bar.set_postfix(loss = '%.3f' % float(loss), 
                                    avg_loss = '%.3f' % float(total_loss / (b_idx + 1)),
                                    # data_time = '%.3f' % float(t_data_1 - t_data_0), 
                                    # net_time = '%.3f' % float(t_net_1 - t_net_0), 
                                    **kwargs)
        t_data_0 = time.time()

    print("avg_loss_over_epoch", total_loss / len(data_loader))

 # Pyten-20201019-AddValidation
def val(net, data_loader, loss_dict, optimizer, scheduler,logger, epoch, metric_dict, use_seg, use_cls, awl):
    net.eval()
    progress_bar = dist_tqdm(data_loader)
    t_data_0 = time.time()
    reset_metrics(metric_dict)
    total_loss = 0
    for b_idx, data_label in enumerate(progress_bar):
        t_data_1 = time.time()
        # reset_metrics(metric_dict)
        global_step = epoch * len(data_loader) + b_idx

        t_net_0 = time.time()
        # pdb.set_trace()
        results = inference(net, data_label, use_seg, use_cls)
        loss = calc_loss(loss_dict, results, logger, global_step, "val", awl)

        scheduler.step(global_step)
        t_net_1 = time.time()

        results = resolve_val_data(results, use_seg, use_cls)

        update_metrics(metric_dict, results)
        if global_step % 20 == 0:
            logger.add_image("val_image", data_label[0][0], global_step=global_step)
            if use_seg:
                seg_color_out = decode_seg_color_map(results["seg_out"][0])
                seg_color_label = decode_seg_color_map( results["seg_label"][0])
                logger.add_image("val_seg/predict", seg_color_out, global_step=global_step, dataformats='HWC')
                logger.add_image("val_seg/label",seg_color_label, global_step=global_step, dataformats='HWC')
            if use_cls:
                cls_color_out = decode_cls_color_map(data_label[0][0], results["cls_out"][0], cfg)
                cls_color_label = decode_cls_color_map(data_label[0][0], results["cls_label"][0], cfg)
                logger.add_image("val_cls/predict", cls_color_out, global_step=global_step, dataformats='HWC')
                logger.add_image("val_cls/label", cls_color_label, global_step=global_step, dataformats='HWC')
                #  results: {'cls_out': cls_out, 'cls_label': cls_label, 'seg_out':seg_out, 'seg_label': seg_label}

            for me_name, me_op in zip(metric_dict['name'], metric_dict['op']):
                logger.add_scalar('val_metric/' + me_name, me_op.get(), global_step=global_step)
        logger.add_scalar('val_meta/lr', optimizer.param_groups[0]['lr'], global_step=global_step)

        if hasattr(progress_bar,'set_postfix'):
            kwargs = {me_name: '%.3f' % me_op.get() for me_name, me_op in zip(metric_dict['name'], metric_dict['op'])}
            progress_bar.set_postfix(loss = '%.3f' % float(loss), 
                                    avg_loss = '%.3f' % float(total_loss / b_idx + 1)
                                    # data_time = '%.3f' % float(t_data_1 - t_data_0), 
                                    # net_time = '%.3f' % float(t_net_1 - t_net_0), 
                                    **kwargs)
        t_data_0 = time.time()
    
    print("avg_loss_over_epoch", total_loss / len(data_loader))
    # Pyten-20201019-SaveBestMetric
    update_best_metric = True
    for me_name, me_op in zip(metric_dict['name'], metric_dict['op']):
        cur_metric = me_op.get()
        if cur_metric < metric_dict["best_metric"][me_name]:
            update_best_metric = False
    if update_best_metric:
        for me_name, me_op in zip(metric_dict['name'], metric_dict['op']):
            metric_dict["best_metric"][me_name] = me_op.get()
                

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    args, cfg = merge_config()

    work_dir = get_work_dir(cfg)

    distributed = False
    if 'WORLD_SIZE' in os.environ:
        distributed = int(os.environ['WORLD_SIZE']) > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
    dist_print(datetime.datetime.now().strftime('[%Y/%m/%d %H:%M:%S]') + ' start training...')
    dist_print(cfg)
    assert cfg.backbone in ['18','34','50','101','152','50next','101next','50wide','101wide']

    train_loader, cls_num_per_lane = get_train_loader(cfg.batch_size, cfg.data_root, cfg.griding_num, cfg.dataset, cfg.use_seg, distributed, cfg.num_lanes, cfg, cfg.only_seg_road)
    if cfg.val:
        val_loader = get_val_loader(cfg.val_batch_size, cfg.data_root, cfg.griding_num, cfg.dataset, cfg.use_seg, distributed, cfg.num_lanes, cfg.only_seg_road)
    
    net = parsingNet(pretrained = True, backbone=cfg.backbone,cls_dim = (cfg.griding_num+1,cls_num_per_lane, cfg.num_lanes),use_seg=cfg.use_seg,use_cls=cfg.use_cls,seg_class_num=cfg.seg_class_num).cuda()
    # Pyten-20201015-AddAutoWeightedLoss
    if "awl" in cfg:
        awl = AutomaticWeightedLoss(cfg.awl)
    else:
        awl = None

    if distributed:
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids = [args.local_rank])
    optimizer = get_optimizer(net, cfg)

    if cfg.finetune is not None:
        dist_print('finetune from ', cfg.finetune)
        state_all = torch.load(cfg.finetune)['model']
        state_clip = {}  # only use backbone parameters
        for k,v in state_all.items():
            if 'model' in k:
                state_clip[k] = v
        net.load_state_dict(state_clip, strict=False)
    if cfg.resume is not None:
        dist_print('==> Resume model from ' + cfg.resume)
        resume_dict = torch.load(cfg.resume, map_location='cpu')
        net.load_state_dict(resume_dict['model'])
        if 'optimizer' in resume_dict.keys():
            optimizer.load_state_dict(resume_dict['optimizer'])
        resume_epoch = int(os.path.split(cfg.resume)[1][2:5]) + 1
    else:
        resume_epoch = 0

    scheduler = get_scheduler(optimizer, cfg, len(train_loader))
    dist_print(len(train_loader))
    metric_dict = get_metric_dict(cfg)
    loss_dict = get_loss_dict(cfg)
    logger = get_logger(work_dir, cfg)
    #cp_projects(work_dir)

    for epoch in range(resume_epoch, cfg.epoch):
        # pdb.set_trace()
        print("epoch:", epoch)
        print("trainging...")
        train(net, train_loader, loss_dict, optimizer, scheduler,logger, epoch, metric_dict, cfg.use_seg, cfg.use_cls, awl)
        
        # Pyten-20201019-AddValidation
        if cfg.val:
            print("validating...")
            val(net, val_loader, loss_dict, optimizer, scheduler,logger, epoch, metric_dict, cfg.use_seg, cfg.use_cls, awl)
        
        save_model(net, optimizer, epoch ,work_dir, distributed)
    if cfg.val:
        for me_name, me_op in zip(metric_dict['name'], metric_dict['op']):
            print(me_name, me_op.get())
    logger.close()
