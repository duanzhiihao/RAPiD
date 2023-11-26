# Model for exporting to ONNX/TENSORRT

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

        self.angle_range = 360.0
        self._build_anchors(input_hw)

    def _build_anchors(self, input_hw):
        # input size (height, width)
        self.input_hw = input_hw
        # sanity check: the input size should be divisible by the largest stride
        imgh, imgw = input_hw
        assert (imgh % self.strides[-1] == 0) and (imgw % self.strides[-1] == 0)

        # Anchors
        anchors = [
            [[18.7807,33.4659], [28.8912,61.7536], [48.6849,68.3897]], # S
            [[45.0668,101.4673], [63.0952,113.5382], [81.3909,134.4554]], # M
            [[91.7364,144.9949], [137.5189,178.4791], [194.4429,250.7985]] # L
        ]
        anchor_shapes = torch.Tensor(anchors).reshape(3, 3, 1, 1, 2)
        self.register_buffer('anchor_shapes', anchor_shapes, persistent=False)
        # first 3 is for three scales (Large, Medium, Small)
        # second 3 is for three anchors per scale
        # (1, 1) will be broadcasted to (nH, nW)
        # the last 2 is for the anchor width and height

    def forward(self, x):
        # x.shape is like (batch, 3, height, width)

        # backbone
        small, medium, large = self.backbone(x)

        # Pyramid networks
        detect_L, feature_L = self.branch_L(large, previous=None)
        detect_M, feature_M = self.branch_M(medium, previous=feature_L)
        detect_S, _ = self.branch_S(small, previous=feature_M)

        # output transformation
        bboxes = torch.cat([
            self.output_transform(detect_S, 0), # S
            self.output_transform(detect_M, 1), # M
            self.output_transform(detect_L, 2) # L
        ], dim=1)
        return bboxes

    def output_transform(self, raw: torch.Tensor, branch_index):
        """ Transform the network output into bounding boxes

        Args:
            raw (torch.Tensor): network output
        """
        device = raw.device

        nB, _, nH, nW = raw.shape
        y_shift = torch.arange(nH, dtype=torch.float, device=device).view(1,1,nH,1)
        x_shift = torch.arange(nW, dtype=torch.float, device=device).view(1,1,1,nW)

        # stride, h dim, w dim, h arange, w arange, anchors
        anchors = self.anchor_shapes[branch_index:branch_index+1]
        nA = anchors.shape[1] # number of anchors, should be 3
        nCH = 6 # number of channels, 6 is (x,y,w,h,angle,conf)

        raw = raw.view(nB, nA, nCH, nH, nW).permute(0, 1, 3, 4, 2).contiguous()
        # now, shape = (nB, nA, nH, nW, nCH)

        # transform xy into image space
        raw[..., 0:2].sigmoid_() # x,y offsets
        imgh, imgw = self.input_hw[0], self.input_hw[1]
        raw[..., 0] = (raw[..., 0] + x_shift) * (imgw / nW) # x in image space
        raw[..., 1] = (raw[..., 1] + y_shift) * (imgh / nH)

        # transform wh into image space
        anchors = anchors.to(device=device)
        raw[..., 2:4] = torch.exp(raw[..., 2:4]) * anchors # (nB, 3, nH, nW, 2) * (1, 3, 1, 1, 2)

        # angle and confidence score
        raw[..., 4:6] = torch.sigmoid(raw[..., 4:6])
        raw[..., 4] = raw[..., 4] * self.angle_range - self.angle_range / 2

        bboxes = raw.view(nB, nA*nH*nW, nCH)
        return bboxes
