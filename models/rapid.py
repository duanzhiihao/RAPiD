import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as tvf

from utils.iou_mask import iou_mask, iou_rle
import models.backbones
import models.losses


class RAPiD(nn.Module):
    def __init__(self, backbone='dark53', **kwargs):
        super().__init__()
        anchors = [
            [18.7807, 33.4659], [28.8912, 61.7536], [48.6849, 68.3897],
            [45.0668, 101.4673], [63.0952, 113.5382], [81.3909, 134.4554],
            [91.7364, 144.9949], [137.5189, 178.4791], [194.4429, 250.7985]
        ]
        indices = [[6,7,8], [3,4,5], [0,1,2]]
        self.anchors_all = torch.Tensor(anchors).float()
        assert self.anchors_all.shape[1] == 2 and len(indices) == 3
        self.index_L = torch.Tensor(indices[0]).long()
        self.index_M = torch.Tensor(indices[1]).long()
        self.index_S = torch.Tensor(indices[2]).long()

        if backbone == 'dark53':
            self.backbone = models.backbones.Darknet53()
            print("Using backbone Darknet-53. Loading ImageNet weights....")
            backbone_imgnet_path = './weights/dark53_imgnet.pth'
            if os.path.exists(backbone_imgnet_path):
                pretrained = torch.load(backbone_imgnet_path)
                self.load_state_dict(pretrained)
            else:
                print('Warning: no ImageNet-pretrained weights found.',
                      'Please check https://github.com/duanzhiihao/RAPiD for it.')
        elif backbone == 'res34':
            self.backbone = models.backbones.resnet34()
        elif backbone == 'res50':
            self.backbone = models.backbones.resnet50()
        elif backbone == 'res101':
            self.backbone = models.backbones.resnet101()
        else:
            raise Exception('Unknown backbone name')
        pnum = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        print('Number of parameters in backbone:', pnum)

        if backbone == 'dark53':
            chS, chM, chL = 256, 512, 1024
        elif backbone in {'res34'}:
            chS, chM, chL = 128, 256, 512
        elif backbone in {'res50','res101'}:
            chS, chM, chL = 512, 1024, 2048
        self.branch_L = models.backbones.YOLOBranch(chL, 18)
        self.branch_M = models.backbones.YOLOBranch(chM, 18, prev_ch=(chL//2,chM//2))
        self.branch_S = models.backbones.YOLOBranch(chS, 18, prev_ch=(chM//2,chS//2))
        
        self.pred_L = PredLayer(self.anchors_all, self.index_L, **kwargs)
        self.pred_M = PredLayer(self.anchors_all, self.index_M, **kwargs)
        self.pred_S = PredLayer(self.anchors_all, self.index_S, **kwargs)

    def forward(self, x, labels=None, **kwargs):
        '''
        x: a batch of images, e.g. shape(8,3,608,608)
        labels: a batch of ground truth
        '''
        assert x.dim() == 4
        self.img_size = x.shape[2:4]

        # go through backbone
        small, medium, large = self.backbone(x)

        # go through detection blocks in three scales
        detect_L, feature_L = self.branch_L(large, previous=None)
        detect_M, feature_M = self.branch_M(medium, previous=feature_L)
        detect_S, _ = self.branch_S(small, previous=feature_M)

        # process the boxes, and calculate loss if there is gt
        boxes_L, loss_L = self.pred_L(detect_L, self.img_size, labels)
        boxes_M, loss_M = self.pred_M(detect_M, self.img_size, labels)
        boxes_S, loss_S = self.pred_S(detect_S, self.img_size, labels)

        if labels is None:
            # assert boxes_L.dim() == 3
            boxes = torch.cat((boxes_L,boxes_M,boxes_S), dim=1)
            return boxes
        else:
            # check all the gt objects are assigned
            gt_num = (labels[:,:,0:4].sum(dim=2) > 0).sum()
            assigned = self.pred_L.gt_num + self.pred_M.gt_num + self.pred_S.gt_num
            assert assigned == gt_num
            self.loss_str = self.pred_L.loss_str + '\n' + self.pred_M.loss_str + \
                            '\n' + self.pred_S.loss_str
            loss = loss_L + loss_M + loss_S
            return loss


class PredLayer(nn.Module):
    '''
    Calculate the output boxes and losses.
    '''
    def __init__(self, all_anchors, anchor_indices, **kwargs):
        super().__init__()
        # self.anchors_all = all_anchors
        self.anchor_indices = anchor_indices
        self.anchors = all_anchors[anchor_indices]
        # anchors: tensor, e.g. shape(2,3), [[116,90],[156,198]]
        self.num_anchors = len(anchor_indices)
        # all anchors, (0, 0, w, h), used for calculating IoU
        self.anch_00wha_all = torch.zeros(len(all_anchors), 5)
        self.anch_00wha_all[:,2:4] = all_anchors # image space, degree

        self.ignore_thre = 0.6

        # self.loss4obj = FocalBCE(reduction='sum')
        self.loss4obj = nn.BCELoss(reduction='sum')
        self.l2_loss = nn.MSELoss(reduction='sum')
        self.bce_loss = nn.BCELoss(reduction='sum')
        loss_angle = kwargs.get('loss_angle', 'period_L1')
        if loss_angle == 'period_L1':
            self.loss4angle = models.losses.period_L1(reduction='sum')
        elif loss_angle == 'period_L2':
            self.loss4angle = models.losses.period_L2(reduction='sum')
        elif loss_angle == 'none':
            # inference
            self.loss4angle = None
        else:
            raise Exception('unknown loss for angle')
        self.laname = loss_angle
        self.angle_range = kwargs.get('angran', 360)
        assert self.angle_range == 360, 'We recommend that angle range = 360'

    def forward(self, raw, img_hw, labels=None):
        """
        Args:
            raw: tensor with shape [batchsize, anchor_num*6, size, size]
            img_hw: int, image resolution
            labels: ground truth annotations        
        """
        assert raw.dim() == 4

        # raw 
        device = raw.device
        nB = raw.shape[0] # batch size
        nA = self.num_anchors # number of anchors
        nH, nW = raw.shape[2:4] # grid size, i.e., feature resolution
        nCH = 6 # number of channels, 6=(x,y,w,h,angle,conf)

        raw = raw.view(nB, nA, nCH, nH, nW)
        raw = raw.permute(0, 1, 3, 4, 2).contiguous()
        # now shape(nB, nA, nH, nW, nCH), meaning (nB x nA x nH x nW) objects

        # sigmoid activation for xy, angle, obj_conf
        xy_offset = torch.sigmoid(raw[..., 0:2]) # x,y
        # linear activation for w, h
        wh_scale = raw[..., 2:4]
        # angle
        if self.laname in {'LL1', 'LL2'}:
            # linear activation
            angle = raw[..., 4]
        else:
            angle = torch.sigmoid(raw[..., 4])
        # logistic activation for objectness confidence
        conf = torch.sigmoid(raw[..., 5])
        # now xy is the offsets, wh are is scaling factor,
        # and angle is normalized between 0~1.

        # calculate pred - xywh obj cls
        x_shift = torch.arange(nW, dtype=torch.float, device=device).view(1,1,1,nW)
        y_shift = torch.arange(nH, dtype=torch.float, device=device).view(1,1,nH,1)

        # NOTE: anchors are not normalized
        anchors = self.anchors.clone().to(device=device)
        anch_w = anchors[:,0].view(1, nA, 1, 1) # image space
        anch_h = anchors[:,1].view(1, nA, 1, 1) # image space

        pred_final = torch.empty(nB, nA, nH, nW, 6, device=device)
        pred_final[..., 0] = (xy_offset[..., 0] + x_shift) / nW # 0-1 space
        pred_final[..., 1] = (xy_offset[..., 1] + y_shift) / nH # 0-1 space
        pred_final[..., 2] = torch.exp(wh_scale[..., 0]) * anch_w # image space
        pred_final[..., 3] = torch.exp(wh_scale[..., 1]) * anch_h # image space
        if self.laname in {'LL1', 'LL2'}:
            pred_final[..., 4] = angle / np.pi * 180 # degree
        else:
            pred_final[..., 4] = angle*self.angle_range - self.angle_range/2 # degree
        pred_final[..., 5] = conf

        if labels is None:
            # inference, convert final predictions to image space
            pred_final[..., 0] *= img_hw[1]
            pred_final[..., 1] *= img_hw[0]
            # self.dt_cache = pred_final.clone()
            return pred_final.view(nB, -1, nCH).detach(), None
        else:
            # training, convert final predictions to be normalized
            pred_final[..., 2] /= img_hw[1]
            pred_final[..., 3] /= img_hw[0]
            # force the normalized w and h to be <= 1
            pred_final[..., 0:4].clamp_(min=0, max=1)

        pred_boxes = pred_final[..., :5].detach() # xywh normalized, a degree
        pred_confs = pred_final[..., 5].detach()

        # target assignment
        obj_mask = torch.zeros(nB, nA, nH, nW, dtype=torch.bool, device=device)
        penalty_mask = torch.ones(nB, nA, nH, nW, dtype=torch.bool, device=device)
        target = torch.zeros(nB, nA, nH, nW, nCH, dtype=torch.float, device=device)

        labels = labels.detach()
        nlabel = (labels[:,:,0:4].sum(dim=2) > 0).sum(dim=1)  # number of objects
        assert (labels[:,:,4].abs() <= 90).all()
        labels = labels.to(device=device)

        tx_all, ty_all = labels[:,:,0] * nW, labels[:,:,1] * nH # 0-nW, 0-nH
        tw_all, th_all = labels[:,:,2], labels[:,:,3] # normalized 0-1
        ta_all = labels[:,:,4] # degree, 0-max_angle

        ti_all = tx_all.long()
        tj_all = ty_all.long()

        # workaround to be compatible with newer version pytorch. See issue #35. 
        img_hw = torch.Tensor(list(img_hw)).to(device=device)

        norm_anch_wh = anchors[:,0:2] / img_hw # normalized
        norm_anch_00wha = self.anch_00wha_all.clone().to(device=device)
        norm_anch_00wha[:,2:4] /= img_hw # normalized

        # traverse all images in a batch
        valid_gt_num = 0
        for b in range(nB):
            n = int(nlabel[b]) # number of ground truths in b'th image
            if n == 0:
                # no ground truth
                continue
            gt_boxes = torch.zeros(n, 5, device=device)
            gt_boxes[:, 2] = tw_all[b, :n] # normalized 0-1
            gt_boxes[:, 3] = th_all[b, :n] # normalized 0-1
            gt_boxes[:, 4] = 0

            # calculate iou between truth and reference anchors
            anchor_ious = iou_mask(gt_boxes, norm_anch_00wha, xywha=True,
                                   mask_size=64, is_degree=True)
            # anchor_ious = iou_rle(gt_boxes, norm_anch_00wha, xywha=True,
            #                       is_degree=True, img_hw=img_hw, normalized=True)
            best_n_all = torch.argmax(anchor_ious, dim=1)
            best_n = best_n_all % self.num_anchors

            valid_mask = torch.zeros(n, dtype=torch.bool, device=device)
            for ind in self.anchor_indices.to(device=device):
                valid_mask = ( valid_mask | (best_n_all == ind) )
            if sum(valid_mask) == 0:
                # no anchor is responsible for any ground truth
                continue
            else:
                valid_gt_num += sum(valid_mask)

            best_n = best_n[valid_mask]
            truth_i = ti_all[b, :n][valid_mask]
            truth_j = tj_all[b, :n][valid_mask]

            gt_boxes[:, 0] = tx_all[b, :n] / nW # normalized 0-1
            gt_boxes[:, 1] = ty_all[b, :n] / nH # normalized 0-1
            gt_boxes[:, 4] = ta_all[b, :n] # degree

            # print(torch.cuda.memory_allocated()/1024/1024/1024, 'GB')
            # gt_boxes e.g. shape(11,4)
            selected_idx = pred_confs[b] > 0.001
            selected = pred_boxes[b][selected_idx]
            if len(selected) < 2000 and len(selected) > 0:
                # ignore some predicted boxes who have high overlap with any groundtruth
                # pred_ious = iou_mask(selected.view(-1,5), gt_boxes, xywha=True,
                #                     mask_size=32, is_degree=True)
                pred_ious = iou_rle(selected.view(-1,5), gt_boxes, xywha=True,
                                    is_degree=True, img_size=tuple(img_hw.tolist()), normalized=True)
                pred_best_iou, _ = pred_ious.max(dim=1)
                to_be_ignored = (pred_best_iou > self.ignore_thre)
                # set mask to zero (ignore) if the pred BB has a large IoU with any gt BB
                penalty_mask[b,selected_idx] = ~to_be_ignored

            penalty_mask[b,best_n,truth_j,truth_i] = 1
            obj_mask[b,best_n,truth_j,truth_i] = 1
            target[b,best_n,truth_j,truth_i,0] = tx_all[b,:n][valid_mask] - tx_all[b,:n][valid_mask].floor()
            target[b,best_n,truth_j,truth_i,1] = ty_all[b,:n][valid_mask] - ty_all[b,:n][valid_mask].floor()
            target[b,best_n,truth_j,truth_i,2] = torch.log(tw_all[b,:n][valid_mask]/norm_anch_wh[best_n,0] + 1e-16)
            target[b,best_n,truth_j,truth_i,3] = torch.log(th_all[b,:n][valid_mask]/norm_anch_wh[best_n,1] + 1e-16)
            # use radian when calculating loss
            target[b,best_n,truth_j,truth_i,4] = gt_boxes[:, 4][valid_mask] / 180 * np.pi
            target[b,best_n,truth_j,truth_i,5] = 1 # objectness confidence

        loss_xy = self.bce_loss(xy_offset[obj_mask], target[...,0:2][obj_mask])
        wh_pred = wh_scale[obj_mask]
        wh_target = target[...,2:4][obj_mask]
        loss_wh = self.l2_loss(wh_pred, wh_target)
        if self.laname in {'LL1', 'LL2'}:
            angle_pred = angle[obj_mask] # radian
        elif self.angle_range == 360:
            angle_pred = angle[obj_mask] * 2 * np.pi - np.pi
        elif self.angle_range == 180:
            angle_pred = angle[obj_mask] * np.pi - np.pi/2
        loss_angle = self.loss4angle(angle_pred, target[..., 4][obj_mask])
        loss_obj = self.loss4obj(conf[penalty_mask], target[...,5][penalty_mask])

        loss = loss_xy + 0.5*loss_wh + loss_angle + loss_obj
        ngt = valid_gt_num + 1e-16
        self.gt_num = valid_gt_num
        self.loss_str = f'level_{nH}x{nW} total {int(ngt)} objects: ' \
                        f'xy/gt {loss_xy/ngt:.3f}, wh/gt {loss_wh/ngt:.3f}' \
                        f', angle/gt {loss_angle/ngt:.3f}, conf {loss_obj:.3f}'

        return None, loss
