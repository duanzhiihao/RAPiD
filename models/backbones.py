import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models


def ConvBnLeaky(in_, out_, k, s):
    '''
    in_: input channel, e.g. 32
    out_: output channel, e.g. 64
    k: kernel size, e.g. 3 or (3,3)
    s: stride, e.g. 1 or (1,1)
    '''
    pad = (k - 1) // 2
    return nn.Sequential(
        nn.Conv2d(in_, out_, k, s, padding=pad, bias=False),
        nn.BatchNorm2d(out_, eps=1e-5, momentum=0.1),
        nn.LeakyReLU(0.1)
    )


class DarkBlock(nn.Module):
    '''
    basic residual block in Darknet53
    in_out: input and output channels
    hidden: channels in the block
    '''
    def __init__(self, in_out, hidden):
        super().__init__()
        self.cbl_0 = ConvBnLeaky(in_out, hidden, k=1, s=1)
        self.cbl_1 = ConvBnLeaky(hidden, in_out, k=3, s=1)

    def forward(self, x):
        residual = x
        x = self.cbl_0(x)
        x = self.cbl_1(x)

        return x + residual


class Darknet53(nn.Module):
    def __init__(self):
        super(Darknet53, self).__init__()
        self.netlist = nn.ModuleList()
        
        # first conv layer
        self.netlist.append(ConvBnLeaky(3, 32, k=3, s=1))

        # Downsampled by 2 (accumulatively), followed by residual blocks
        self.netlist.append(ConvBnLeaky(32, 64, k=3, s=2))
        for _ in range(1):
            self.netlist.append(DarkBlock(in_out=64, hidden=32))

        # Downsampled by 4 (accumulatively), followed by residual blocks
        self.netlist.append(ConvBnLeaky(64, 128, k=3, s=2))
        for _ in range(2):
            self.netlist.append(DarkBlock(in_out=128, hidden=64))
        
        # Downsampled by 8 (accumulatively), followed by residual blocks
        self.netlist.append(ConvBnLeaky(128, 256, k=3, s=2))
        for _ in range(8):
            self.netlist.append(DarkBlock(in_out=256, hidden=128))
        assert len(self.netlist) == 15

        # Downsampled by 16 (accumulatively), followed by residual blocks
        self.netlist.append(ConvBnLeaky(256, 512, k=3, s=2))
        for _ in range(8):
            self.netlist.append(DarkBlock(in_out=512, hidden=256))
        assert len(self.netlist) == 24

        # Downsampled by 32 (accumulatively), followed by residual blocks
        self.netlist.append(ConvBnLeaky(512, 1024, k=3, s=2))
        for _ in range(4):
            self.netlist.append(DarkBlock(in_out=1024, hidden=512))
        assert len(self.netlist) == 29
        # end creating Darknet-53 back bone layers

    def forward(self, x):
        for i in range(0,15):
            x = self.netlist[i](x)
        small = x
        for i in range(15,24):
            x = self.netlist[i](x)
        medium = x
        for i in range(24,29):
            x = self.netlist[i](x)
        large = x

        return small, medium, large


class ResNetBackbone(nn.Module):
    '''
    Args:
        tv_model: torch vision model
    '''
    def __init__(self, tv_model):
        super().__init__()
        self.conv1 = tv_model.conv1
        self.bn1 = tv_model.bn1
        self.relu = tv_model.relu
        self.maxpool = tv_model.maxpool

        self.layer1 = tv_model.layer1
        self.layer2 = tv_model.layer2
        self.layer3 = tv_model.layer3
        self.layer4 = tv_model.layer4
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        small = self.layer2(x)
        medium = self.layer3(small)
        large = self.layer4(medium)
        return small, medium, large

def resnet34():
    print('Using backbone ResNet-34. Loading ImageNet weights...')
    model = torchvision.models.resnet34(pretrained=True)
    return ResNetBackbone(model)

def resnet50():
    print('Using backbone ResNet-50. Loading ImageNet weights...')
    model = torchvision.models.resnet50(pretrained=True)
    return ResNetBackbone(model)

def resnet101():
    print('Using backbone ResNet-101. Loading ImageNet weights...')
    model = torchvision.models.resnet101(pretrained=True)
    return ResNetBackbone(model)


class YOLOBranch(nn.Module):
    '''
    Args:
        in_: int, input channel number
        out_: int, output channel number, typically = 3 * 6 [x,y,w,h,a,conf]
        has_previous: bool, True if this is not the first detection layer
        prev_ch: (int,int), the Conv2d channel for the previous feature,
                 default: None
    '''
    # def __init__(self, in_, out_=18, has_previous=False, prev_ch=None):
    def __init__(self, in_, out_=18, prev_ch=None):
        super(YOLOBranch, self).__init__()
        assert in_ % 2 == 0, 'input channel must be divisible by 2'

        # tmp_ch = prev_ch if prev_ch is not None else (in_, in_//2)
        if prev_ch:
            self.process = ConvBnLeaky(prev_ch[0], prev_ch[1], k=1, s=1)
            in_after_cat = in_ + prev_ch[1]
        else:
            self.process = None
            in_after_cat = in_

        self.cbl_0 = ConvBnLeaky(in_after_cat, in_//2, k=1, s=1)
        self.cbl_1 = ConvBnLeaky(in_//2, in_, k=3, s=1)

        self.cbl_2 = ConvBnLeaky(in_, in_//2, k=1, s=1)
        self.cbl_3 = ConvBnLeaky(in_//2, in_, k=3, s=1)

        self.cbl_4 = ConvBnLeaky(in_, in_//2, k=1, s=1)
        self.cbl_5 = ConvBnLeaky(in_//2, in_, k=3, s=1)

        self.to_box = nn.Conv2d(in_, out_, kernel_size=1, stride=1)
        
    def forward(self, x, previous=None):
        '''
        Args:
            x: feature from backbone, for large/medium/small size
            previous: feature from the lower spatial resolution
        '''
        if previous is not None:
            pre = self.process(previous)
            pre = F.interpolate(pre, scale_factor=2, mode='nearest')
            x = torch.cat((pre, x), dim=1)
        
        x = self.cbl_0(x)
        x = self.cbl_1(x)
        x = self.cbl_2(x)
        x = self.cbl_3(x)
        feature = self.cbl_4(x)
        x = self.cbl_5(feature)
        detection = self.to_box(x)

        return detection, feature
