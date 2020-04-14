import numpy as np
from PIL import Image
import random
from collections import defaultdict
import torch
from torchvision import transforms

from utils.iou_mask import iou_mask, iou_geometry, iou_rle


def normalize_bbox(xywha, w, h, max_angle=1):
    '''
    normalize bounding boxes to 0~1 range
    xywha: torch.tensor, shape(5) or shape(N,5)
    '''
    assert torch.is_tensor(xywha)

    if xywha.dim() == 1:
        assert xywha.shape[0] == 5
        xywha[0] /= w
        xywha[1] /= h
        xywha[2] /= w
        xywha[3] /= h
        xywha[4] /= max_angle
    elif xywha.dim() == 2:
        assert xywha.shape[1] == 5
        xywha[:,0] /= w
        xywha[:,1] /= h
        xywha[:,2] /= w
        xywha[:,3] /= h
        xywha[:,4] /= max_angle
    else:
        raise Exception('unkown bbox format')
    
    return xywha


def denormalize_bbox(xywha, w, h, angle_factor):
    '''
    denormalize bounding boxes to image range
    xywha: torch.tensor, shape(5) or shape(N,5)
    '''
    assert torch.is_tensor(xywha)

    if xywha.dim() == 1:
        assert xywha.shape[0] == 5
        xywha[0] *= w
        xywha[1] *= h
        xywha[2] *= w
        xywha[3] *= h
        xywha[4] *= angle_factor
    elif xywha.dim() == 2:
        assert xywha.shape[1] == 5
        xywha[:,0] *= w
        xywha[:,1] *= h
        xywha[:,2] *= w
        xywha[:,3] *= h
        xywha[:,4] *= angle_factor
    else:
        raise Exception('unkown bbox format')
    
    return xywha


def rect_to_square(img, labels, target_size, pad_value=0, aug=False):
    '''
    Arguments:
    img: PIL image
    labels: torch.tensor, shape(N,5), not normalized [x,y,w,h,a]
    target_size: int, e.g. 608
    pad_value: int
    aug: bool
    '''
    assert isinstance(img, Image.Image)
    ori_h, ori_w = img.height, img.width

    # resize to target input size (usually smaller)
    resize_scale = target_size / max(ori_w,ori_h)
    # unpad_w, unpad_h = target_size * w / max(w,h), target_size * h / max(w,h)
    unpad_w, unpad_h = int(ori_w*resize_scale), int(ori_h*resize_scale)
    img = transforms.functional.resize(img, (unpad_h,unpad_w))

    # pad to square
    if aug:
        # random placing
        left = random.randint(0, target_size - unpad_w)
        top = random.randint(0, target_size - unpad_h)
    else:
        left = (target_size - unpad_w) // 2
        top = (target_size - unpad_h) // 2
    right = target_size - unpad_w - left
    bottom = target_size - unpad_h - top

    img = transforms.functional.pad(img, padding=(left,top,right,bottom), fill=0)
    # record the padding info
    img_tl = (left, top) # start of the true image
    img_wh = (unpad_w, unpad_h)

    # modify labels
    if labels is not None:
        labels[:,0:4] *= resize_scale
        labels[:,0] += left
        labels[:,1] += top
    
    pad_info = torch.Tensor((ori_w, ori_h) + img_tl + img_wh)
    return img, labels, pad_info


def detection2original(boxes, pad_info):
    '''
    recover the bbox labels from in square to in original rectangle
    Args:
    boxes: tensor, not normalized, rows of [x,y,w,h,a]
    pad_info: (ori w, ori h, tl x, tl y, imw, imh)
    '''
    assert boxes.dim() == 2
    ori_w, ori_h, tl_x, tl_y, imw, imh = pad_info
    boxes[:,0] = (boxes[:,0] - tl_x) / imw * ori_w
    boxes[:,1] = (boxes[:,1] - tl_y) / imh * ori_h
    boxes[:,2] = boxes[:,2] / imw * ori_w
    boxes[:,3] = boxes[:,3] / imh * ori_h

    return boxes


# def MSC(detections, is_degree=True, dist_thres=0.04, **kwargs):
#     '''
#     Mean shift clustering method for bounding box refinement. Proposed in: \
#     Omnidirectional Pedestrian Detection by Rotation Invariant Training, WACV2019

#     Args:
#         detections: rows of (x,y,w,h,angle,conf,...)
#     '''
#     assert detections.dim() == 2 and detections.shape[1] >= 6
#     img_size = kwargs.get('img_size', 1)
#     img_h,img_w = img_size,img_size if isinstance(img_size, int) else img_size
#     if len(detections) <= 1:
#         return detections
#     # normalize the x & y in detections
#     dts = detections.clone()
#     dts[:,0] /= img_w
#     dts[:,1] /= img_h

#     # mean shift
#     while True:
#         dists = distance(dts, dts)
#         new_dts = dts.clone()
#         for i, ds in enumerate(dists):
#             mask = (ds < dist_thres)
#             num = mask.sum()
#             new_dts[i,0] = dts[mask,0].sum() / num
#             new_dts[i,1] = dts[mask,1].sum() / num
#         if torch.isclose(new_dts, dts).all():
#             break
#         dts = new_dts
#     dts[:,0] *= img_w
#     dts[:,1] *= img_h

#     # merge
#     results = []
#     remaining = torch.ones(dts.shape[0], dtype=torch.bool)
#     scores = dts[:,5]
#     for i in range(dts.shape[0]):
#         if not remaining[i]:
#             continue
#         dists = distance(dts[i,:], dts).squeeze()
#         cluster_mask = (dists < dist_thres)
#         cluster_dts = dts[cluster_mask,:]
#         weighted = scores[cluster_mask].unsqueeze(1)
#         new = dts[i,:]
#         new[:4] = (weighted*cluster_dts[:,:4]).sum(dim=0) / weighted.sum()
#         max_s, idx = torch.max(weighted, dim=0)
#         new[4] = cluster_dts[idx, 4] # angle
#         new[5] = max_s # confidence

#         remaining[cluster_mask] = False
#         results.append(new)

#     return torch.stack(results, dim=0)


def MSC(detections, is_degree=True, dist_thres=0.04, **kwargs):
    '''
    Mean shift clustering method for bounding box refinement. Proposed in: \
    Omnidirectional Pedestrian Detection by Rotation Invariant Training, WACV2019

    Args:
        detections: rows of (x,y,w,h,angle,conf,...)
    '''
    assert detections.dim() == 2 and detections.shape[1] >= 6
    img_size = kwargs.get('img_size', 1)
    img_h,img_w = img_size,img_size if isinstance(img_size, int) else img_size
    if len(detections) <= 1:
        return detections
    # normalize the x & y in detections
    detections_n = detections.clone()
    detections_n[:,0] /= img_w
    detections_n[:,1] /= img_h

    dts = detections_n.clone()

    # mean shift
    while True:
        dists = distance(dts, detections_n)
        new_dts = dts.clone()
        for i, ds in enumerate(dists):
            mask = (ds < dist_thres)
            num = mask.sum()
            new_dts[i,0] = detections_n[mask,0].sum() / num
            new_dts[i,1] = detections_n[mask,1].sum() / num
        if torch.isclose(new_dts, dts).all():
            break
        dts = new_dts

    # Find the clusters
    clusters = torch.zeros(dts.shape[0])
    cluster_id = 0
    for i in range(dts.shape[0]):
        if clusters[i]:
            continue
        dists = distance(dts[i,:], dts).squeeze()
        cluster_mask = (dists < dist_thres)
        cluster_candidate, idx = torch.max(clusters[cluster_mask], dim=0)
        if cluster_candidate: # if one of the neighbors already clustered
            clusters[cluster_mask] = cluster_candidate
        else:
            cluster_id += 1
            clusters[cluster_mask] = cluster_id

    # Merge the clusters
    results = []
    num_clusters, idx = torch.max(clusters, dim=0)
    for cluster_id in range(1, num_clusters.int()+1):
        cluster_dts = dts[clusters==cluster_id,:]
        weights = cluster_dts[:, 5].unsqueeze(1)
        new = dts[0,:].clone()
        new[:4] = (weights*cluster_dts[:,:4]).sum(dim=0) / weights.sum()

        #!!TO-DO!! Angle should be completed for the new center coordinates
        # new[4:5] = (weights*cluster_dts[:,4:5]).sum(dim=0) / weights.sum()
        new[4] = _xy2radius(new[:2])

        max_s, idx = torch.max(weights, dim=0)
        new[5] = max_s # confidence
        results.append(new)

    results = torch.stack(results, dim=0)
    results[:,0] *= img_w
    results[:,1] *= img_h

    return results


def distance(boxes1, boxes2):
    '''
    Calculate euclidean distances between the centers of boxes1 and boxes2

    Arguments:
        boxes1: tensor or numpy, shape(N,>=2), second dimension=(cx, cy, ...)
        boxes2: tensor or numpy, shape(M,>=2), second dimension=(cx, cy, ...)

    Return:
        iou_matrix: tensor, shape(N,M), float32, the calculated distances
    '''
    assert torch.is_tensor(boxes1) and torch.is_tensor(boxes2)
    if boxes1.dim() == 1:
        boxes1 = boxes1.unsqueeze(0)
    assert boxes1.dim() == 2 and boxes2.dim() == 2
    assert boxes1.shape[1] >= 2 and boxes2.shape[1] >= 2
    
    xy1 = boxes1[:,0:2].view(-1,1,2)
    xy2 = boxes2[:,0:2].view(1,-1,2)
    ds = (xy1 - xy2).pow(2).sum(dim=-1).sqrt()
    return ds


def nms(detections, is_degree=True, nms_thres=0.45, majority=None, **kwargs):
    '''
    Non-maximum suppression for bounding boxes with angle AND without category
    
    Args:
        detections: rows of (x,y,w,h,angle,conf,...)
        is_degree: True if input angle is in degree
        majority (optional): int, a BB is suppresssed if the number of votes \
        less than majority. default: None
    '''
    assert (detections.dim() == 2) and (detections.shape[1] >= 6)
    device = detections.device
    if detections.shape[0] == 0:
        return detections
    # sort by confidence
    idx = torch.argsort(detections[:,5], descending=True)
    detections = detections[idx,:]

    boxes = detections[:,0:5] # only [x,y,w,h,a]
    valid = torch.zeros(boxes.shape[0], dtype=torch.bool, device=device)
    # the first one is always valid
    valid[0] = True
    # only one candidate at the beginning. Its votes number is 1 (it self)
    votes = [1]
    for i in range(1, boxes.shape[0]):
        # compute IoU with valid boxes
        # ious = iou_mask(boxes[i], boxes[valid,:], True, 32, is_degree=is_degree)
        ious = iou_rle(boxes[i], boxes[valid,:], xywha=True, is_degree=is_degree,
                      img_size=2048)
        # the i'th BB is invalid if it is similar to any valid BB
        if (ious >= nms_thres).any():
            if majority is not None:
                # take down the votes for majority voting
                vote_idx = torch.argmax(ious).item()
                votes[vote_idx] += 1
            continue
        # else, this box is valid
        valid[i] = True
        # the votes number of the new candidate BB is 1 (it self)
        votes.append(1)

    selected = detections[valid,:]
    if majority is None:
        # standard NMS
        return selected
    votes_valid = (torch.Tensor(votes) >= majority)
    return selected[votes_valid,:]


# def voting_suppression(detections, majority, is_degree=True, nms_thres=0.45):
#     '''
#     Args:
#         detections: rows of (x,y,w,h,angle,conf,...)
#         majority: a BB is suppresssed if the number of vote < majority
#         is_degree: True if input angle is in degree
#     '''
#     assert (detections.dim() == 2) and (detections.shape[1] >= 6)
#     device = detections.device
#     if detections.shape[0] == 0:
#         return detections
#     # sort by confidence
#     idx = torch.argsort(detections[:,5], descending=True)
#     detections = detections[idx,:]

#     boxes = detections[:,0:5] # only [x,y,w,h,a]
#     valid = torch.zeros(boxes.shape[0], dtype=torch.bool, device=device)
#     # the first one is always valid
#     valid[0] = True
#     for i in range(1, boxes.shape[0]):
#         # compute IoU with valid boxes
#         # iou = iou_mask(boxes[i], boxes[valid,:], True, 32, is_degree=is_degree)
#         iou = iou_rle(boxes[i], boxes[valid,:], xywha=True, is_degree=is_degree,
#                       img_size=2048)
#         # if similar to any valid box, this box is invalid
#         if (iou >= nms_thres).any():
#             continue
#         # else, this box is valid
#         valid[i] = 1
    
#     return detections[valid,:]


def seq_nms(seq_dts, temporal_iou=0.5, spatial_iou=0.45):
    '''
    seq-NMS

    Warning: this function will modify the seq_dts IN-PLACE

    Args:
        seq_dts: list of 2-dim tensors, each tensor shape (# of bbox, x y w h a conf)
    '''
    assert len(seq_dts) != 0, 'seq_dts is an empty list'
    nf = len(seq_dts)
    if nf == 1:
        # if there is only one frame, perform standard NMS on it
        return [nms(seq_dts[0], nms_thres=spatial_iou)]

    scores = seq_dts[0][:,5]
    paths = []
    for i in range(nf-1):
        # shape: (num of bboxes, 6)
        prev_bbs = seq_dts[i]
        cur_bbs = seq_dts[i+1]
        assert prev_bbs.shape[1] == cur_bbs.shape[1] == 6, 'invalid bounding box shape'
        if cur_bbs.shape[0] == 0 or prev_bbs.shape[0] == 0:
            raise NotImplementedError() # TODO

        ious = iou_mask(cur_bbs[:5], prev_bbs[:5], xywha=True, mask_size=32)
        valid_mask = (ious > temporal_iou)
        
        # previous cumulative scores
        prev_scores = prev_bbs[:,5].clone().unsqueeze(0)
        prev_scores = prev_scores * valid_mask
        max_score, max_idx = torch.max(prev_scores, dim=1)
        cur_bbs[:,5] += max_score
        paths.append(max_idx)


def seq_nms_causal(cur_dts, prev_dts, temporal_iou=0.5, momentum=None):
    # shape: (num of bboxes, x y w h a conf num)
    assert prev_dts.shape[1] == 7 # x, y, w, h, a, conf, tublet_len
    if cur_dts.shape[1] == 6:
        cur_dts = torch.cat([cur_dts, torch.ones(cur_dts.shape[0],1)], dim=1)
    else:
        assert cur_dts.shape[1] == 7, 'shape of current detections must be (:, 6 or 7)'
    assert (cur_dts[:,6] == 1).all()

    if cur_dts.shape[0] == 0 or prev_dts.shape[0] == 0:
        return cur_dts

    ious = iou_mask(cur_dts[:,:5], prev_dts[:,:5], xywha=True, mask_size=32)
    valid_mask = (ious > temporal_iou)
    
    # previous cumulative scores
    prev_scores = prev_dts[:,5].clone().unsqueeze(0)
    prev_scores = prev_scores * valid_mask
    max_score, max_idx = torch.max(prev_scores, dim=1)
    cur_dts = cur_dts.clone()
    
    valid_mask = (torch.sum(valid_mask, dim=1) >= 1)
    if momentum is None:
        prev_len = prev_dts[max_idx,6] * valid_mask
        # cur_dts[:,5] = (cur_dts[:,5] + max_score*prev_len) / (prev_len+1)
        cur_dts[:,5] = cur_dts[:,5] + max_score
    else:
        assert momentum > 0 and momentum < 1
        max_score = max_score * valid_mask
        cur_dts[:,5] = momentum * max_score + (1 - momentum) * cur_dts[:,5]
    
    cur_dts[valid_mask,6] += prev_dts[max_idx,6][valid_mask]
    return cur_dts


class Seq_NMS():
    def __init__(self, buffer, temporal_iou=0.5, spatial_iou=0.3):
        raise NotImplementedError()
'''
Notes
causal seems to work: maintain the max sum link of the previous frame
only look at the previous frame, do not look at future
do this first, easy to implement

look at both past and future 10 frames

maintain a max
new link: compare with max, update max
end path: compare with max, delete path
'''


def _xy2radius(prediction, return_degree=True):
    '''
    use x,y coordinates in normalized image to calculate reference angle \
    regarding to the image center.

    Args:
        prediction: torch.tensor, the last dimension should be [x,y,something]
    
    Return:
        reference: torch.tensor, same shape with xs and ys, degree
    '''
    assert return_degree == True
    assert prediction.shape[-1] >= 2
    xs = prediction[..., 0] - 0.5
    ys = -(prediction[..., 1] - 0.5)
    reference = torch.atan(xs/ys) / np.pi * 180 # -90 ~ 90

    if torch.isnan(reference).any():
        print('warning: some of reference angle is nan')
        reference[torch.isnan(reference)] = 0
    assert not torch.isinf(reference).any()

    return reference


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
