import torch, os, cv2
import pdb
# from model.model import parsingNet
# from model.model_fcn import parsingNet
from utils.common import merge_config
from utils.dist_utils import dist_print
import torch
import time
import scipy.special, tqdm
import numpy as np
import torchvision.transforms as transforms
from data.dataset import LaneTestDataset
from data.constant import culane_row_anchor, tusimple_row_anchor
from utils.common import decode_seg_color_map

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    args, cfg = merge_config()

    print("model", args.model)
    if args.model == "F":
        from model.model_fcn import parsingNet
    elif args.model == "M":
        from model.resnet_mtan import parsingNet
    else:
        from model.model import parsingNet

    dist_print('start testing...')
    assert cfg.backbone in ['18','34','50','101','152','50next','101next','50wide','101wide']

    if cfg.cls_dataset == 'CULane':
        cls_num_per_lane = 18
        lane_num = 4
        row_anchor = culane_row_anchor
    elif cfg.cls_dataset == 'Bdd100k':
        cls_num_per_lane = 56#18
        lane_num = 4#14
        row_anchor = tusimple_row_anchor
    elif cfg.cls_dataset == 'Tusimple':
        cls_num_per_lane = 56
        lane_num = 4
        row_anchor = tusimple_row_anchor
    else:
        raise NotImplementedError

    #net = parsingNet(pretrained = False, backbone=cfg.backbone,cls_dim = (cfg.griding_num+1,cls_num_per_lane, lane_num),
    #                use_seg=cfg.use_seg).cuda() # we dont need auxiliary segmentation in testing
    net = parsingNet(pretrained = True, backbone=cfg.backbone,cls_dim = (cfg.griding_num+1,cls_num_per_lane, cfg.num_lanes),
                    use_seg=cfg.use_seg,use_cls=cfg.use_cls,seg_class_num=cfg.seg_class_num).cuda()
    state_dict = torch.load(cfg.test_model, map_location='cpu')['model']
    compatible_state_dict = {}
    # pdb.set_trace()
    for k, v in state_dict.items():
        if 'module.' in k:
            compatible_state_dict[k[7:]] = v
        else:
            compatible_state_dict[k] = v

    net.load_state_dict(compatible_state_dict, strict=False)
    net.eval()

    img_transforms = transforms.Compose([
        # transforms.Resize((288, 800)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    # if cfg.dataset == 'CULane':
    # #if cfg.dataset == 'CULane':
    #     # splits = ['test0_normal.txt', 'test1_crowd.txt', 'test2_hlight.txt', 'test3_shadow.txt', 'test4_noline.txt', 'test5_arrow.txt', 'test6_curve.txt', 'test7_cross.txt', 'test8_night.txt']
    #     # datasets = [LaneTestDataset(cfg.data_root,os.path.join(cfg.data_root, 'list/test_split/'+split),img_transform = img_transforms) for split in splits]
    #     # img_w, img_h = 1640, 590
    #     row_anchor = culane_row_anchor
    # elif cfg.dataset == 'Bdd100k':
    #     row_anchor = tusimple_row_anchor#culane_row_anchor
    # elif cfg.dataset == 'Tusimple':
    #     # splits = ['test.txt']
    #     # datasets = [LaneTestDataset(cfg.data_root,os.path.join(cfg.data_root, split),img_transform = img_transforms) for split in splits]
    #     # img_w, img_h = 1280, 720
    #     row_anchor = tusimple_row_anchor
    # else:
    #     raise NotImplementedError
    #for split, dataset in zip(splits, datasets):
    #    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle = False, num_workers=1)
        # fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        # print(split[:-3]+'avi')
        # vout = cv2.VideoWriter(split[:-3]+'avi', fourcc , 30.0, (img_w, img_h))
    #    for i, data in enumerate(tqdm.tqdm(loader)):
            # imgs, names = data
    # pdb.set_trace()
    #data_folder = os.path.join(os.getcwd(), "testdata")
    data_folder = "/home/pantengteng/datasets/customed_lane_data"
    # data_folder = "/nfs/nas/dataset/test_set_operational_scenarios/traffic_light"
    save_folder = os.path.join(os.getcwd(), f"ufld_test_results/test_new_result/{cfg.cls_dataset}") #_SH
    # save_folder = os.path.join(os.getcwd(), f"test_result/{cfg.dataset}") #_SH
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    img_list = os.listdir(data_folder)
    total_time = 0
    for img_name in img_list:
        # img_name = "962.jpg"
        org_img = cv2.imread(os.path.join(data_folder, img_name))
        img_h, img_w, _ = org_img.shape
        img =cv2.resize(org_img, (800, 288))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # In_img = torch.from_numpy(img).cuda()
        # Totensor转换为CHW.permute(0,3,1,2)
        in_img = img_transforms(img).unsqueeze(0).cuda()

        start_time = time.time()
        with torch.no_grad():
            if cfg.use_seg:
                cls_out, seg_out = net(in_img)
            else:
                cls_out = net(in_img)
            end_time = time.time()
        total_time += end_time - start_time

        col_sample = np.linspace(0, 800 - 1, cfg.griding_num)
        col_sample_w = col_sample[1] - col_sample[0]
        out_j = cls_out[0].data.cpu().numpy()
        out_j = out_j[:, ::-1, :]
        prob = scipy.special.softmax(out_j[:-1, :, :], axis=0)
        idx = np.arange(cfg.griding_num) + 1
        idx = idx.reshape(-1, 1, 1)
        loc = np.sum(prob * idx, axis=0)
        out_j = np.argmax(out_j, axis=0)
        loc[out_j == cfg.griding_num] = 0
        out_j = loc

        # import pdb; pdb.set_trace()
        # vis = cv2.imread(os.path.join(cfg.data_root,names[0]))
        for i in range(out_j.shape[1]):
            if np.sum(out_j[:, i] != 0) > 2:
                for k in range(out_j.shape[0]):
                    if out_j[k, i] > 0:
                        ppp = (int(out_j[k, i] * col_sample_w * img_w / 800) - 1, int(img_h * (row_anchor[cls_num_per_lane-1-k]/288)) - 1 )
                        cv2.circle(org_img, ppp,5,(0,255,0),-1)
        cv2.imwrite(os.path.join(save_folder, cfg.save_prefix + img_name), org_img)

        # Pyten-20201020-AddSegOut
        if cfg.use_seg:
            # pdb.set_trace()
            seg_predict = torch.argmax(seg_out, dim=1).squeeze(0)
            seg_img = decode_seg_color_map(seg_predict)
            seg_img = seg_img.data.cpu().numpy()
            seg_img =cv2.resize(seg_img, (img_w, img_h))
            cv2.imwrite(os.path.join(save_folder, cfg.save_prefix +"seg"+ img_name), seg_img)

    print(total_time / len(img_list))