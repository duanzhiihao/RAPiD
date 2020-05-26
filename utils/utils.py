from PIL import Image
import random
import torch
from torchvision import transforms

from utils.iou_mask import iou_mask, iou_rle


def normalize_bbox(xywha, w, h, max_angle=1):
    '''
    Normalize bounding boxes to 0~1 range

    Args:
        xywha: torch.tensor, bounding boxes, shape(...,5)
        w: image width
        h: image height
        max_angle: the angle will be divided by max_angle
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


def rect_to_square(img, labels, target_size, pad_value=0, aug=False):
    '''
    Pre-processing during training and testing

    1. Resize img such that the longer side of the image = target_size;
    2. Pad the img it to square

    Arguments:
        img: PIL image
        labels: torch.tensor, shape(N,5), [cx, cy, w, h, angle], not normalized
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
    Recover the bbox from the resized and padded image to the original image.

    Args:
        boxes: tensor, rows of [cx, cy, w, h, angle]
        pad_info: (ori w, ori h, tl x, tl y, imw, imh)
    '''
    assert boxes.dim() == 2
    ori_w, ori_h, tl_x, tl_y, imw, imh = pad_info
    boxes[:,0] = (boxes[:,0] - tl_x) / imw * ori_w
    boxes[:,1] = (boxes[:,1] - tl_y) / imh * ori_h
    boxes[:,2] = boxes[:,2] / imw * ori_w
    boxes[:,3] = boxes[:,3] / imh * ori_h

    return boxes


def nms(detections, is_degree=True, nms_thres=0.45, img_size=2048):
    '''
    Single-class non-maximum suppression for bounding boxes with angle.
    
    Args:
        detections: rows of (x,y,w,h,angle,conf,...)
        is_degree: True -> input angle is degree, False -> radian
        nms_thres: suppresion IoU threshold
        img_size: int, preferably the image size
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
                      img_size=img_size)
        # the i'th BB is invalid if it is similar to any valid BB
        if (ious >= nms_thres).any():
            continue
        # else, this box is valid
        valid[i] = True
        # the votes number of the new candidate BB is 1 (it self)
        votes.append(1)

    selected = detections[valid,:]
    return selected
