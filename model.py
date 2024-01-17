
import torch.nn as nn
import math
import torch
from torch.nn import functional as F

from ops import Conv2d
from attention import CDCM
from attention import ChannelAttention

#将xception的卷积换成CDD卷积与vc卷积的混合，包括深度可分离卷积，其余的不换
class SeparableConv2d(nn.Module):
    def __init__(self, pdc,in_channels, out_channels, kernel_size, stride, padding, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        # 逐通道卷积：groups=in_channels=out_channels
        self.cnv_cdc_pointwise = Conv2d_cd(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1,
                                   bias=bias)
        # 逐点卷积：普通1x1卷积
        self.cnv_pdc=Conv2d(pdc=pdc,in_channels=in_channels,out_channels=in_channels,kernel_size=kernel_size,stride=stride,padding=padding,
                            dilation=dilation,groups=in_channels,bias=bias)


    def forward(self, x):
        # x = self.conv1(x)
        x=self.cnv_pdc(x)
        # x = self.pointwise(x)
        x=self.cnv_cdc_pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self,pdc, in_filters, out_filters, reps, strides=1, start_with_relu=True, grow_first=True):
        #:parm reps:块重复次数
        super(Block, self).__init__()

        # Middle flow无需做这一步，而其余块需要，以做跳连
        # 1）Middle flow输入输出特征图个数始终一致，且stride恒为1
        # 2）其余块需要stride=2，这样可以将特征图尺寸减半，获得与最大池化减半特征图尺寸同样的效果
        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(in_filters, out_filters, kernel_size=1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        rep = []

        filters = in_filters
        if grow_first:
            rep.append(self.relu)
            # 这里的卷积不改变特征图尺寸
            rep.append(SeparableConv2d(pdc,in_filters, out_filters, kernel_size=3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps - 1):
            rep.append(self.relu)
            # 这里的卷积不改变特征图尺寸
            rep.append(SeparableConv2d(pdc,filters, filters, kernel_size=3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(self.relu)
            # 这里的卷积不改变特征图尺寸
            rep.append(SeparableConv2d(pdc,in_filters, out_filters, kernel_size=3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        # Middle flow 的stride恒为1，因此无需做池化，而其余块需要
        # 其余块的stride=2，因此这里的最大池化可以将特征图尺寸减半
        if strides != 1:
            rep.append(nn.MaxPool2d(kernel_size=3, stride=strides, padding=1))
        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x += skip
        return x


class Conv2d_cd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.8):

        super(Conv2d_cd, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        out_normal = self.conv(x)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal
        else:
            [C_out,C_in, kernel_size,kernel_size] = self.conv.weight.shape
            kernel_diff = self.conv.weight.sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None]
            out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0, groups=self.conv.groups)

# theta 控制着差分卷积及Vanilla卷积的贡献，值越大意味着gradient clue占比越重
            return out_normal - self.theta*out_diff

class MFusionModule(nn.Module):
    def __init__(self, in_chan=728*4, out_chan=728, *args, **kwargs):
        super(MFusionModule, self).__init__()
        self.convblk = nn.Sequential(
            Conv2d_cd(in_chan, out_chan, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_chan),
            nn.ReLU()
        )

        self.ca = ChannelAttention(out_chan, ratio=8)


        self.init_weight()

    def forward(self, x1, x2,x3,x4):
        fuse_fea = self.convblk(torch.cat((x1, x2,x3,x4), dim=1))
        # print("通道注意的输入")
        # print(fuse_fea.shape)
        fuse_fea = fuse_fea + fuse_fea * self.ca(fuse_fea)
        # print("通道注意的输出")
        # print(self.ca(fuse_fea).shape)
        return fuse_fea

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None:
                    nn.init.constant_(ly.bias, 0)

class MCDC_Net(nn.Module):
    def __init__(self, pdcs,num_classes=2):
        super(MCDC_Net, self).__init__()
        self.num_classes = num_classes  # 总分类数
        self.cdcm2 = CDCM(728, 728)
        self.fusion = MFusionModule()
        self.x1 = Conv2d_cd(in_channels=64, out_channels=728, kernel_size=1, stride=8, padding=0, bias=False)


        ################################## 定义 Entry flow ###############################################################
        self.cnv_cdc1 = Conv2d_cd(in_channels=3, out_channels=32, kernel_size=1, stride=2, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.cnv_cdc2 = Conv2d_cd(in_channels=32, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        # do relu here

        # Block中的参数顺序：in_filters,out_filters,reps,stride,start_with_relu,grow_first
        self.block1 = Block(pdcs[0],64, 128, 2, 2, start_with_relu=False, grow_first=True)
        self.block2 = Block(pdcs[1],128, 256, 2, 2, start_with_relu=True, grow_first=True)
        self.block3 = Block(pdcs[2],256, 728, 2, 2, start_with_relu=True, grow_first=True)

        ################################### 定义 Middle flow ############################################################
        self.block4 = Block(pdcs[3],728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block5 = Block(pdcs[4],728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block6 = Block(pdcs[5],728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block7 = Block(pdcs[6],728, 728, 3, 1, start_with_relu=True, grow_first=True)

        self.block8 = Block(pdcs[7],728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block9 = Block(pdcs[8],728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block10 = Block(pdcs[9],728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block11 = Block(pdcs[10],728, 728, 3, 1, start_with_relu=True, grow_first=True)

        #################################### 定义 Exit flow ###############################################################
        self.block12 = Block(pdcs[11],728, 1024, 2, 2, start_with_relu=True, grow_first=False)

        self.conv3 = SeparableConv2d(pdcs[0],1024, 1536, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(1536)

        # do relu here
        self.conv4 = SeparableConv2d(pdcs[1],1536, 2048, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(2048)

        self.fc = nn.Linear(2048, num_classes)
        ###################################################################################################################

        # ------- init weights --------
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        # -----------------------------

    def forward(self, x):
        ################################## 定义 Entry flow ###############################################################
        x=self.cnv_cdc1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x=self.cnv_cdc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x1=self.x1(x)
        x1=self.cdcm2(x1)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x2=self.cdcm2(x)

        ################################### 定义 Middle flow ############################################################

        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x3=self.cdcm2(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x4=self.cdcm2(x)
        x = self.fusion(x1, x2, x3, x4)

        #################################### 定义 Exit flow ###############################################################
        x = self.block12(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
