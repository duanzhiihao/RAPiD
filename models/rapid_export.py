# A temporary workaround for exporting to ONNX/TENSORRT

import torch
import torch.nn as nn

from models.backbones import Darknet53, YOLOBranch


class RAPiD(nn.Module):
    def __init__(self, input_hw=(1024,1024)):
        super().__init__()
        # Backbone network
        self.backbone = Darknet53()
        self.strides = [8, 16, 32] # S, M, L

        # Feature pyramid network blocks
        chS, chM, chL = 256, 512, 1024
        self.branch_L = YOLOBranch(in_=chL, out_=18)
        self.branch_M = YOLOBranch(in_=chM, out_=18, prev_ch=(chL//2,chM//2))
        self.branch_S = YOLOBranch(in_=chS, out_=18, prev_ch=(chM//2,chS//2))

        # Anchors
        anchors = [
            [[18.7807,33.4659], [28.8912,61.7536], [48.6849,68.3897]], # S
            [[45.0668,101.4673], [63.0952,113.5382], [81.3909,134.4554]], # M
            [[91.7364,144.9949], [137.5189,178.4791], [194.4429,250.7985]] # L
        ]
        # self.anchors: [anchor_L, anchor_M, anchor_S]
        self.angle_range = 360.0
        self.input_hw = input_hw
        self._build_branch_infos(anchors)

    def _build_branch_infos(self, anchors):
        imgh, imgw = self.input_hw
        assert (imgh % self.strides[-1] == 0) and (imgw % self.strides[-1] == 0)

        branch_infos = []
        for i, strd in enumerate(self.strides):
            # check image size is multiple of stride
            assert (imgh % strd == 0) and (imgw % strd == 0)
            # feature dimension
            nH, nW = imgh // strd, imgw // strd
            # mesh grid
            y_shift = torch.arange(nH, dtype=torch.float).view(1,1,nH,1)
            x_shift = torch.arange(nW, dtype=torch.float).view(1,1,1,nW)
            # anchors
            anch = torch.Tensor(anchors[i]).reshape(1, 3, 1, 1, 2)
            # stride, h dim, w dim, h arange, w arange
            info = (strd, nH, nW, y_shift, x_shift, anch)
            branch_infos.append(info)
        self._branch_infos = branch_infos

    def forward(self, x, labels=None):
        '''
        x: a batch of images, e.g. shape(8,3,608,608)
        labels: a batch of ground truth
        '''
        assert x.shape[2:4] == self.input_hw

        # backbone
        small, medium, large = self.backbone(x)

        # Pyramid networks in three spatial scales
        detect_L, feature_L = self.branch_L(large, previous=None)
        detect_M, feature_M = self.branch_M(medium, previous=feature_L)
        detect_S, _ = self.branch_S(small, previous=feature_M)

        if labels is None:
            # testing
            assert not self.training
            bboxes = torch.cat([
                self.output_transform(detect_S, 0), # S
                self.output_transform(detect_M, 1), # M
                self.output_transform(detect_L, 2) # L
            ], dim=1)
            return bboxes
        else:
            # training
            raise NotImplementedError()

    def output_transform(self, raw: torch.Tensor, branch_index=None):
        """ Transform the network output into bounding boxes in image space

        Args:
            raw (torch.Tensor): network output
        """
        # assert not raw.requires_grad, 'the output_transform() function is for testing only'
        assert branch_index is not None

        device = raw.device
        # stride, h dim, w dim, h arange, w arange, anchors
        s, nH, nW, y_shift, x_shift, anchors = self._branch_infos[branch_index]
        nA = anchors.shape[1] # number of anchors, should be 3
        nCH = 6 # number of channels, 6=(x,y,w,h,angle,conf)
        nB = raw.shape[0] # batch size
        assert raw.shape == (nB, nA*nCH, nH, nW)

        raw = raw.view(nB, nA, nCH, nH, nW)
        raw = raw.permute(0, 1, 3, 4, 2).contiguous()
        # now shape(nB, nA, nH, nW, nCH), meaning (nB x nA x nH x nW) objects

        # transform xy into image space
        raw[..., 0:2].sigmoid_() # x,y offsets
        imgh, imgw = self.input_hw
        y_shift, x_shift = y_shift.to(device=device), x_shift.to(device=device)
        raw[..., 0].add_(x_shift).mul_(imgw / nW) # x in image space
        raw[..., 1].add_(y_shift).mul_(imgh / nH) # y in image space

        # transform wh into image space
        anchors = anchors.to(device=device)
        raw[..., 2:4].exp_().mul_(anchors) # (nB, 3, nH, nW, 2) * (1, 3, 1, 1, 2)

        # angle and confidence score
        raw[..., 4:6].sigmoid_()
        raw[..., 4].mul_(self.angle_range).sub_(self.angle_range/2) # angle from 0~1 to -180~180

        bboxes = raw.view(nB, nA*nH*nW, nCH)
        return bboxes
