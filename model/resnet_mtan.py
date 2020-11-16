import torch
import torch.nn as nn
import torch.nn.functional as F
import  numpy as np
import sys
sys.path.append("./model")
import resnet

from resnet_dilated import ResnetDilated
from aspp import DeepLabHead
from resnet import Bottleneck, conv1x1

class parsingNet(torch.nn.Module):
    def __init__(self, size=(288, 800), pretrained=True, backbone='50', cls_dim=(37, 10, 4), use_seg=False, use_cls=True, seg_class_num=19, dilate_scale=8):
        super(parsingNet, self).__init__()

        self.size = size
        self.w = size[1]
        self.h = size[0]
        self.cls_dim = cls_dim # (num_gridding, num_cls_per_lane, num_of_lanes)
        # num_cls_per_lane is the number of row anchors
        self.use_seg = use_seg
        self.use_cls = use_cls
        self.total_dim = np.prod(cls_dim)
        self.class_num = seg_class_num
        self.cls_size = int(self.h * self.w * 8 / (dilate_scale * dilate_scale))

        if backbone in ['18', '34']:
            ch = [64, 128, 256, 512]
        else:
            #for resnet50,101...
            ch = [256, 512, 1024, 2048]

        backbone_net = ResnetDilated(resnet.__dict__['resnet' + backbone](pretrained=True), dilate_scale)
        
        self.tasks = ['seg', 'lane']
        self.num_out_channels = {'seg': seg_class_num, 'lane': cls_dim[2]}
        
        self.shared_conv = nn.Sequential(backbone_net.conv1, backbone_net.bn1, backbone_net.relu1, backbone_net.maxpool)

        # We will apply the attention over the last bottleneck layer in the ResNet. 
        self.shared_layer1_b = backbone_net.layer1[:-1] 
        self.shared_layer1_t = backbone_net.layer1[-1]

        self.shared_layer2_b = backbone_net.layer2[:-1]
        self.shared_layer2_t = backbone_net.layer2[-1]

        self.shared_layer3_b = backbone_net.layer3[:-1]
        self.shared_layer3_t = backbone_net.layer3[-1]

        self.shared_layer4_b = backbone_net.layer4[:-1]
        self.shared_layer4_t = backbone_net.layer4[-1]

        # Define task specific attention modules using a similar bottleneck design in residual block
        # (to avoid large computations)
        self.encoder_att_1 = nn.ModuleList([self.att_layer(ch[0], ch[0] // 4, ch[0]) for _ in self.tasks])
        self.encoder_att_2 = nn.ModuleList([self.att_layer(2 * ch[1], ch[1] // 4, ch[1]) for _ in self.tasks])
        self.encoder_att_3 = nn.ModuleList([self.att_layer(2 * ch[2], ch[2] // 4, ch[2]) for _ in self.tasks])
        self.encoder_att_4 = nn.ModuleList([self.att_layer(2 * ch[3], ch[3] // 4, ch[3]) for _ in self.tasks])

        # Define task shared attention encoders using residual bottleneck layers
        # We do not apply shared attention encoders at the last layer,
        # so the attended features will be directly fed into the task-specific decoders.
        self.encoder_block_att_1 = self.conv_layer(ch[0], ch[1] // 4)
        self.encoder_block_att_2 = self.conv_layer(ch[1], ch[2] // 4)
        self.encoder_block_att_3 = self.conv_layer(ch[2], ch[3] // 4)
        
        self.down_sampling = nn.MaxPool2d(kernel_size=2, stride=2)

        # Define task-specific decoders using ASPP modules
        # self.decoders = nn.ModuleList([DeepLabHead(ch[-1], self.num_out_channels[t]) for t in self.tasks])
        self.pool = torch.nn.Conv2d(512,8,1) if backbone in ['34','18'] else torch.nn.Conv2d(2048,8,1)
        self.cls = torch.nn.Sequential(
                torch.nn.Linear(self.cls_size, 2048),
                torch.nn.ReLU(),
                nn.Dropout(p=0.3),
                torch.nn.Linear(2048, self.total_dim),
            )

        self.decoder = DeepLabHead(ch[-1], self.num_out_channels["seg"])


    def forward(self, x):
        # Shared convolution
        x = self.shared_conv(x)
        
        # Shared ResNet block 1
        u_1_b = self.shared_layer1_b(x)
        u_1_t = self.shared_layer1_t(u_1_b)

        # Shared ResNet block 2
        u_2_b = self.shared_layer2_b(u_1_t)
        u_2_t = self.shared_layer2_t(u_2_b)

        # Shared ResNet block 3
        u_3_b = self.shared_layer3_b(u_2_t)
        u_3_t = self.shared_layer3_t(u_3_b)
        
        # Shared ResNet block 4
        u_4_b = self.shared_layer4_b(u_3_t)
        u_4_t = self.shared_layer4_t(u_4_b)

        # Attention block 1 -> Apply attention over last residual block
        a_1_mask = [att_i(u_1_b) for att_i in self.encoder_att_1]  # Generate task specific attention map
        a_1 = [a_1_mask_i * u_1_t for a_1_mask_i in a_1_mask]  # Apply task specific attention map to shared features
        a_1 = [self.down_sampling(self.encoder_block_att_1(a_1_i)) for a_1_i in a_1]
        
        # Attention block 2 -> Apply attention over last residual block
        a_2_mask = [att_i(torch.cat((u_2_b, a_1_i), dim=1)) for a_1_i, att_i in zip(a_1, self.encoder_att_2)]
        a_2 = [a_2_mask_i * u_2_t for a_2_mask_i in a_2_mask]
        a_2 = [self.encoder_block_att_2(a_2_i) for a_2_i in a_2]
        
        # Attention block 3 -> Apply attention over last residual block
        a_3_mask = [att_i(torch.cat((u_3_b, a_2_i), dim=1)) for a_2_i, att_i in zip(a_2, self.encoder_att_3)]
        a_3 = [a_3_mask_i * u_3_t for a_3_mask_i in a_3_mask]
        a_3 = [self.encoder_block_att_3(a_3_i) for a_3_i in a_3]
        
        # Attention block 4 -> Apply attention over last residual block (without final encoder)
        a_4_mask = [att_i(torch.cat((u_4_b, a_3_i), dim=1)) for a_3_i, att_i in zip(a_3, self.encoder_att_4)]
        a_4 = [a_4_mask_i * u_4_t for a_4_mask_i in a_4_mask]
        
        # Task specific decoders
        # out = [0 for _ in self.tasks]
        # for i, t in enumerate(self.tasks):
        #     out[i] = F.interpolate(self.decoders[i](a_4[i]), size=out_size, mode='bilinear', align_corners=True)
        #     if t == 'segmentation':
        #         out[i] = F.log_softmax(out[i], dim=1)
        #     if t == 'normal':
        #         out[i] = out[i] / torch.norm(out[i], p=2, dim=1, keepdim=True)
        # return out
        seg_out = self.decoder(a_4[0])
        seg_out = F.interpolate(seg_out,scale_factor = 8,mode='bilinear')
        fea = self.pool(a_4[1]).view(-1, self.cls_size)
        cls_out = self.cls(fea).view(-1, *self.cls_dim)

        if self.use_seg and self.use_cls:
            return cls_out, seg_out
        elif self.use_seg:
            return seg_out
        else:
            return cls_out
    
    def att_layer(self, in_channel, intermediate_channel, out_channel):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=intermediate_channel, kernel_size=1, padding=0),
            nn.BatchNorm2d(intermediate_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=intermediate_channel, out_channels=out_channel, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channel),
            nn.Sigmoid())
        
    def conv_layer(self, in_channel, out_channel):
        downsample = nn.Sequential(conv1x1(in_channel, 4 * out_channel, stride=1),
                                   nn.BatchNorm2d(4 * out_channel))
        return Bottleneck(in_channel, out_channel, downsample=downsample)
