import torch, os, datetime
import numpy as np

import pdb
# from model.model import parsingNet
from model.model_fcn import parsingNet
from data.dataloader_multiset import get_seg_train_loader, get_cls_train_loader, get_seg_val_loader, get_cls_val_loader

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

        # Pyten-20201023-AddLossFiter
        if data_src[0] not in results:
            continue

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


def train(net, seg_loader, cls_loader, loss_dict, optimizer, scheduler, logger, epoch, metric_dict, use_seg, use_cls, awl, iters_per_ep, cfg):
    net.train()
    progress_bar = dist_tqdm(list(range(iters_per_ep)))
    t_data_0 = time.time()
    # Pyten-20201019-FixBug
    reset_metrics(metric_dict)
    total_loss = 0
    for idx in progress_bar:
        
        train_seg_iter = enumerate(seg_loader)
        train_cls_iter = enumerate(cls_loader)
        try:
            _, seg_data = next(train_seg_iter)
        except:
            train_seg_iter = enumerate(seg_loader)
            seg_data = next(train_seg_iter)
        
        try:
            _, cls_data = next(train_cls_iter)
        except:
            _, train_cls_iter = enumerate(cls_loader)
            _, cls_data = next(train_cls_iter)
        
        t_data_1 = time.time()
        # reset_metrics(metric_dict)
        global_step = epoch * iters_per_ep + idx

        t_net_0 = time.time()
        # pdb.set_trace()

        seg_img, seg_label = seg_data
        seg_img, seg_label = seg_img.cuda(), seg_label.cuda()

        cls_img, cls_label = cls_data
        cls_img, cls_label = cls_img.cuda(), cls_label.cuda()

        cls_out, _ = net(cls_img)
        cls_results = {'cls_out': cls_out, 'cls_label': cls_label}

        optimizer.zero_grad()
        loss_cls = calc_loss(loss_dict, cls_results, logger, global_step, "train", awl)
        loss_cls.backward()

        _, seg_out = net(seg_img)
        seg_results = {'seg_out':seg_out, 'seg_label': seg_label}

        loss_seg = calc_loss(loss_dict, seg_results, logger, global_step, "train", awl)
        loss_seg.backward()
        
        total_loss = total_loss + loss_cls + loss_seg
        
        optimizer.step()
        scheduler.step(global_step)
        t_net_1 = time.time()
         
        results = dict(cls_results, **seg_results)
        results = resolve_val_data(results, use_seg, use_cls)

        update_metrics(metric_dict, results)
        if global_step % 20 == 0:
            # Pyten-20201012-AddImage2TBD
            logger.add_image("train_seg/image", seg_img[0], global_step=global_step)
            if use_seg:
                seg_color_out = decode_seg_color_map(results["seg_out"][0])
                seg_color_label = decode_seg_color_map( results["seg_label"][0])
                logger.add_image("train_seg/predict", seg_color_out, global_step=global_step, dataformats='HWC')
                logger.add_image("train_seg/label",seg_color_label, global_step=global_step, dataformats='HWC')
            
            logger.add_image("train_cls/image", cls_img[0], global_step=global_step)
            if use_cls:
                cls_color_out = decode_cls_color_map(cls_img[0], results["cls_out"][0], cfg)
                cls_color_label = decode_cls_color_map(cls_img[0], results["cls_label"][0], cfg)
                logger.add_image("train_cls/predict", cls_color_out, global_step=global_step, dataformats='HWC')
                logger.add_image("train_cls/label", cls_color_label, global_step=global_step, dataformats='HWC')
            #  results: {'cls_out': cls_out, 'cls_label': cls_label, 'seg_out':seg_out, 'seg_label': seg_label}

            for me_name, me_op in zip(metric_dict['name'], metric_dict['op']):
                logger.add_scalar('train_metric/' + me_name, me_op.get(), global_step=global_step)
        logger.add_scalar('train/meta/lr', optimizer.param_groups[0]['lr'], global_step=global_step)

        if hasattr(progress_bar,'set_postfix'):
            kwargs = {me_name: '%.3f' % me_op.get() for me_name, me_op in zip(metric_dict['name'], metric_dict['op'])}
            progress_bar.set_postfix(loss_seg = '%.3f' % float(loss_seg), 
                                    loss_cls = '%.3f' % float(loss_cls), 
                                    avg_loss = '%.3f' % float(total_loss / (idx + 1)),
                                    # data_time = '%.3f' % float(t_data_1 - t_data_0), 
                                    # net_time = '%.3f' % float(t_net_1 - t_net_0), 
                                    **kwargs)
        t_data_0 = time.time()

    print("avg_loss_over_epoch", total_loss / len(data_loader))

 # Pyten-20201019-AddValidation
def val(net, val_seg_loader, val_cls_loader, loss_dict, optimizer, scheduler,logger, epoch, metric_dict, use_seg, use_cls, awl, cfg):
    net.eval()
    # validate segmentation
    progress_bar = dist_tqdm(val_seg_loader)
    reset_metrics(metric_dict)
    seg_loss = 0
    for s_idx, seg_data in enumerate(progress_bar):

        seg_global_step = epoch * len(val_seg_loader) + s_idx

        seg_img, seg_label = seg_data
        seg_img, seg_label = seg_img.cuda(), seg_label.cuda()
        cls_out, seg_out = net(seg_img)
        seg_results = {'seg_out':seg_out, 'seg_label': seg_label}
        
        loss_seg = calc_loss(loss_dict, seg_result, logger, seg_global_step, "val_seg", awl)
        seg_loss += loss_seg
        optimizer.zero_grad()
        loss_seg.backward()
        optimizer.step()
        scheduler.step(seg_global_step)

        results = resolve_val_data(seg_results, use_seg=True)

        update_metrics(metric_dict, results)
        if seg_global_step % 20 == 0:
            logger.add_image("val_seg/image", seg_img[0], global_step=seg_global_step)
            if use_seg:
                seg_color_out = decode_seg_color_map(results["seg_out"][0])
                seg_color_label = decode_seg_color_map( results["seg_label"][0])
                logger.add_image("val_seg/seg_predict", seg_color_out, global_step=seg_global_step, dataformats='HWC')
                logger.add_image("val_seg/seg_label",seg_color_label, global_step=seg_global_step, dataformats='HWC')
            if use_cls:
                cls_out = torch.argmax(cls_out, dim=1)
                cls_color_out = decode_cls_color_map(seg_img, cls_out[0], cfg)
                logger.add_image("val_seg/cls_predict", cls_color_out, global_step=seg_global_step, dataformats='HWC')

            for me_name, me_op in zip(metric_dict['name'], metric_dict['op']):
                data_src = metric_dict['data_src'][i]
                # Pyten-20201023-AddMetricFilter
                if data_src[0] not in results:
                    continue
                logger.add_scalar('val_seg_metric/' + me_name, me_op.get(), global_step=seg_global_step)

        logger.add_scalar('val_seg_meta/lr', optimizer.param_groups[0]['lr'], global_step=seg_global_step)

        if hasattr(progress_bar,'set_postfix'):
            kwargs = {me_name: '%.3f' % me_op.get() for me_name, me_op in zip(metric_dict['name'], metric_dict['op'])}
            progress_bar.set_postfix(loss = '%.3f' % float(loss_seg), 
                                    avg_loss = '%.3f' % float(seg_loss / (s_idx + 1))
                                    # data_time = '%.3f' % float(t_data_1 - t_data_0), 
                                    # net_time = '%.3f' % float(t_net_1 - t_net_0), 
                                    **kwargs)

    # validate lane detection
    progress_bar = dist_tqdm(val_cls_loader)
    cls_loss = 0
    for c_idx, cls_data in enumerate(progress_bar):

        cls_global_step = epoch * len(val_cls_loader) + c_idx

        cls_img, cls_label = cls_data
        cls_img, cls_label = cls_img.cuda(), cls_label.cuda()
        cls_out, seg_out = net(cls_img)
        cls_results = {'cls_out':cls_out, 'cls_label': cls_label}
        
        loss_cls += calc_loss(loss_dict, cls_result, logger, cls_global_step, "val_cls", awl)
        cls_loss += loss_cls
        optimizer.zero_grad()
        loss_cls.backward()
        optimizer.step()
        scheduler.step(cls_global_step)

        results = resolve_val_data(cls_results, use_cls=True)

        update_metrics(metric_dict, results)
        if cls_global_step % 20 == 0:
            logger.add_image("val_cls/image", cls_img[0], global_step=cls_global_step)
            if use_seg:
                seg_out = torch.argmax(cls_out, dim=1)
                seg_color_out = decode_seg_color_map(seg_out[0])
                logger.add_image("val_cls/seg_predict", seg_color_out, global_step=cls_global_step, dataformats='HWC')
            if use_cls:
                cls_color_out = decode_cls_color_map(cls_img, cls_out[0], cfg)
                cls_color_label = decode_cls_color_map(cls_img, cls_label[0], cfg)
                logger.add_image("val_cls/cls_predict", cls_color_out, global_step=cls_global_step, dataformats='HWC')
                logger.add_image("val_cls/cls_label", cls_color_label, global_step=cls_global_step, dataformats='HWC')

            for me_name, me_op in zip(metric_dict['name'], metric_dict['op']):
                data_src = metric_dict['data_src'][i]
                # Pyten-20201023-AddMetricFilter
                if data_src[0] not in results:
                    continue
                logger.add_scalar('val_metric/' + me_name, me_op.get(), global_step=cls_global_step)

        logger.add_scalar('val_meta/lr', optimizer.param_groups[0]['lr'], global_step=cls_global_step)

        if hasattr(progress_bar,'set_postfix'):
            kwargs = {me_name: '%.3f' % me_op.get() for me_name, me_op in zip(metric_dict['name'], metric_dict['op'])}
            progress_bar.set_postfix(loss = '%.3f' % float(loss_cls), 
                                    avg_loss = '%.3f' % float(cls_loss / (c_idx + 1))
                                    # data_time = '%.3f' % float(t_data_1 - t_data_0), 
                                    # net_time = '%.3f' % float(t_net_1 - t_net_0), 
                                    **kwargs)
    
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

    train_cls_loader, cls_num_per_lane = get_cls_train_loader(cfg.batch_size, cfg.cls_data_root, cfg.griding_num, cfg.cls_dataset, distributed, cfg.num_lanes, cfg)
    train_seg_loader = get_seg_train_loader(cfg.batch_size, cfg.seg_data_root, cfg.seg_dataset, distributed)
    if cfg.val:
        val_cls_loader = get_cls_val_loader(cfg.val_batch_size, cfg.cls_data_root, cfg.griding_num, cfg.cls_dataset, distributed, cfg.num_lanes, cfg)
        val_seg_loader = get_seg_val_loader(cfg.val_batch_size, cfg.seg_data_root, cfg.seg_dataset, distributed)

    # net = parsingNet(pretrained = True, backbone=cfg.backbone,cls_dim = (cfg.griding_num+1,cls_num_per_lane, cfg.num_lanes),use_seg=cfg.use_seg,use_cls=cfg.use_cls).cuda()
    net = parsingNet(pretrained = True, backbone=cfg.backbone,cls_dim = (cfg.griding_num+1,cls_num_per_lane, cfg.num_lanes),use_seg=cfg.use_seg).cuda()
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

    scheduler = get_scheduler(optimizer, cfg, cfg.iters_per_ep)
    dist_print(cfg.iters_per_ep)
    metric_dict = get_metric_dict(cfg)
    loss_dict = get_loss_dict(cfg)
    logger = get_logger(work_dir, cfg)
    #cp_projects(work_dir)

    for epoch in range(resume_epoch, cfg.epoch):
        # pdb.set_trace()
        print("epoch:", epoch)
        print("trainging with {} seg data and cls {} data...".format(len(train_seg_loader), len(train_cls_loader)))
        train(net, train_seg_loader, train_cls_loader, loss_dict, optimizer, scheduler,logger, epoch, metric_dict, cfg.use_seg, cfg.use_cls, awl, cfg.iters_per_ep, cfg)
        
        # Pyten-20201019-AddValidation
        if cfg.val:
            print("validating with {} seg_data and{} cls data...".format(len(val_seg_loader), len(val_cls_loader)))
            val(net, val_seg_loader, val_cls_loader, loss_dict, optimizer, scheduler, logger, epoch, metric_dict, cfg.use_seg, cfg.use_cls, awl, cfg)
        
        save_model(net, optimizer, epoch ,work_dir, distributed)
    if cfg.val:
        for me_name, me_op in zip(metric_dict['name'], metric_dict['op']):
            print(me_name, me_op.get())
    logger.close()
