from math import pi
import torch

import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="1"


def bboxes_iou(bboxes_a, bboxes_b, xyxy=True):
    """Calculate the Intersection of Unions (IoUs) between bounding boxes.
    IoU is calculated as a ratio of area of the intersection
    and area of the union.
    Args:
        bbox_a (array): An array whose shape is :math:`(N, 4)`.
            :math:`N` is the number of bounding boxes.
            The dtype should be :obj:`numpy.float32`.
        bbox_b (array): An array similar to :obj:`bbox_a`,
            whose shape is :math:`(K, 4)`.
            The dtype should be :obj:`numpy.float32`.
    Returns:
        array:
        An array whose shape is :math:`(N, K)`. \
        An element at index :math:`(n, k)` contains IoUs between \
        :math:`n` th bounding box in :obj:`bbox_a` and :math:`k` th bounding \
        box in :obj:`bbox_b`.
    from: https://github.com/chainer/chainercv
    """
    assert bboxes_a.dim() == bboxes_b.dim() == 2
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError

    # top left
    if xyxy:
        tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
        # bottom right
        br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
    else:
        tl = torch.max((bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
                        (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2))
        # bottom right
        br = torch.min((bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
                        (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2))

        area_a = torch.prod(bboxes_a[:, 2:], 1)
        area_b = torch.prod(bboxes_b[:, 2:], 1)
    en = (tl < br).type(tl.type()).prod(dim=2)
    area_i = torch.prod(br - tl, 2) * en  # * ((tl < br).all())
    return area_i / (area_a[:, None] + area_b - area_i)

    
def xywha2vertex(box, is_degree, stack=True):
    '''
    Args:
        box: tensor, shape(batch,5), 5=(x,y,w,h,a), xy is center,
             angle is radian

    Return:
        tensor, shape(batch,4,2): topleft, topright, br, bl
    '''
    assert is_degree == False and box.dim() == 2 and box.shape[1] >= 5
    batch = box.shape[0]
    device = box.device

    center = box[:,0:2]
    w = box[:,2]
    h = box[:,3]
    rad = box[:,4]

    # calculate two vector
    verti = torch.empty((batch,2), dtype=torch.float32, device=device)
    verti[:,0] = (h/2) * torch.sin(rad)
    verti[:,1] = - (h/2) * torch.cos(rad)

    hori = torch.empty(batch,2, dtype=torch.float32, device=device)
    hori[:,0] = (w/2) * torch.cos(rad)
    hori[:,1] = (w/2) * torch.sin(rad)


    tl = center + verti - hori
    tr = center + verti + hori
    br = center - verti + hori
    bl = center - verti - hori

    if not stack:
        return torch.cat([tl,tr,br,bl], dim=1)
    return torch.stack((tl,tr,br,bl), dim=1)


def vertex2masks(vertices, mask_size=128):
    '''
    Arguments:
        vertices: tensor, shape(batch,4,2)
                  4 means 4 corners of the box, in 0~1 normalized range
                    top left [x,y],
                    top right [x,y],
                    bottom right [x,y],
                    bottom left [x,y]
        mask_size: int or (h,w), size of the output tensor

    Return:
        tensor, shape(batch,size,size), 0/1 mask of the bounding box
    '''
    # assert (vertices >= 0).all() and (vertices <= 1).all()
    assert vertices.dim() == 3 and vertices.shape[1:3] == (4,2)
    device = vertices.device
    batch = vertices.shape[0]
    mh,mw = (mask_size,mask_size) if isinstance(mask_size,int) else mask_size

    # create meshgrid
    gx = torch.linspace(0,1,steps=mw, device=device).view(1,1,-1)
    gy = torch.linspace(0,1,steps=mh, device=device).view(1,-1,1)

    # for example batch=9, all the following shape(9,1,1)
    tl_x = vertices[:,0,0].view(-1,1,1)
    tl_y = vertices[:,0,1].view(-1,1,1)
    tr_x = vertices[:,1,0].view(-1,1,1)
    tr_y = vertices[:,1,1].view(-1,1,1)
    br_x = vertices[:,2,0].view(-1,1,1)
    br_y = vertices[:,2,1].view(-1,1,1)
    bl_x = vertices[:,3,0].view(-1,1,1)
    bl_y = vertices[:,3,1].view(-1,1,1)

    # # x1y1=tl, x2y2=tr
    # top = (tr_y-tl_y)*gx + (tl_x-tr_x)*gy + tl_y*tr_x - tr_y*tl_x < 0
    # # x1y1=tr, x2y2=br
    # right = (br_y-tr_y)*gx + (tr_x-br_x)*gy + tr_y*br_x - br_y*tr_x < 0
    # # x1y1=br, x2y2=bl
    # botom = (bl_y-br_y)*gx + (br_x-bl_x)*gy + br_y*bl_x - bl_y*br_x < 0
    # # x1y1=bl, x2y2=tl
    # left = (tl_y-bl_y)*gx + (bl_x-tl_x)*gy + bl_y*tl_x - tl_y*bl_x < 0
    # mask = top * right * botom * left

    # x1y1=tl, x2y2=tr
    mask = (tr_y-tl_y)*gx + (tl_x-tr_x)*gy + tl_y*tr_x - tr_y*tl_x < 0
    # x1y1=tr, x2y2=br
    mask *= (br_y-tr_y)*gx + (tr_x-br_x)*gy + tr_y*br_x - br_y*tr_x < 0
    # x1y1=br, x2y2=bl
    mask *= (bl_y-br_y)*gx + (br_x-bl_x)*gy + br_y*bl_x - bl_y*br_x < 0
    # x1y1=bl, x2y2=tl
    mask *= (tl_y-bl_y)*gx + (bl_x-tl_x)*gy + bl_y*tl_x - tl_y*bl_x < 0

    assert mask.shape == (batch,mh,mw)
    return mask


def iou_pairs_mask(boxes1, boxes2, xywha, mask_size=128, is_degree=True):
    '''
    use mask method to calculate IOU between corresponding boxes

    Arguments:
        boxes0, boxes1: tensor, shape(Batch,5), 5=(x, y, w, h, angle0~90)
                           two boxes must have the same shape
        is_degree: True if degree, False if radian
        mask_size: int, the size of mask tensor, larger -> more precise but slower

    Return:
        tensor, shape(Batch,), float32, ious
    '''
    assert torch.is_tensor(boxes1) and torch.is_tensor(boxes2)
    assert boxes1.shape == boxes2.shape
    # assert (boxes1[:,4] >= 0).all()
    assert xywha == True
    device = boxes1.device
    batch = boxes1.shape[0]

    if is_degree:
        # assert (boxes1[:,4] <= 180).all() and (boxes2[:,4] <= 180).all()
        # convert to radian
        boxes1[:,4] = boxes1[:,4] * pi / 180
        boxes2[:,4] = boxes2[:,4] * pi / 180
    else:
        # radian
        # assert (boxes1[:,4] <= pi).all() and (boxes2[:,4] <= pi).all()
        pass

    # get vertices, [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    vts1 = xywha2vertex(boxes1, is_degree=False) # shape(Batch,4,2)
    vts2 = xywha2vertex(boxes2, is_degree=False) # shape(Batch,4,2)

    x = torch.cat((vts1[:,:,0],vts2[:,:,0]), dim=1) # shape(Batch,8)
    y = torch.cat((vts1[:,:,1],vts2[:,:,1]), dim=1) # shape(Batch,8)

    # select the smallest area that contain two boxes
    xmin, _ = x.min(dim=1) # shape(Batch,)
    xmax, _ = x.max(dim=1) # shape(Batch,)
    ymin, _ = y.min(dim=1) # shape(Batch,)
    ymax, _ = y.max(dim=1) # shape(Batch,)

    area = torch.empty(batch,1,2, device=device) # shape(Batch,1,2), 2=[w, h]
    area[:,0,0] = xmax - xmin
    area[:,0,1] = ymax - ymin

    # top-left vertex, shape(Batch,1,2)
    topleft = torch.stack((xmin,ymin), dim=1).unsqueeze(dim=1)
    # coordinates in original to coordinate in small area
    vts1 = (vts1 - topleft) / area # shape(Batch,4,2)
    vts2 = (vts2 - topleft) / area # shape(Batch,4,2)

    # calculate two maskes
    mask1 = vertex2masks(vts1, mask_size=mask_size) # shape(Batch,size,size)
    mask2 = vertex2masks(vts2, mask_size=mask_size) # shape(Batch,size,size)

    inter = mask1 * mask2
    union = (mask1 + mask2) > 0

    # for debug
    if False:
        # visualize one of the IOU pairs
        import matplotlib.pyplot as plt
        from random import randint
        for _ in range(5):
            i = randint(0,batch-1)

            x1,y1,w1,h1 = boxes1[i,:4]
            a1 = boxes1[i,4] * 180 / pi
            print(f'box 1, x: {x1}, y: {y1}, w: {w1}, h: {h1}, a: {a1}')
            x2,y2,w2,h2 = boxes2[i,:4]
            a2 = boxes2[i,4] * 180 / pi
            print(f'box 2, x: {x2}, y: {y2}, w: {w2}, h: {h2}, a: {a2}')

            plt.subplot(2,2,1)
            plt.title('box1')
            plt.axis('off')
            plt.imshow(mask1[i].numpy().astype('float32'),cmap='gray')
            plt.subplot(2,2,2)
            plt.title('box2')
            plt.axis('off')
            plt.imshow(mask2[i].numpy().astype('float32'),cmap='gray')
            plt.subplot(2,2,3)
            plt.title('intersection')
            plt.axis('off')
            plt.imshow(inter[i].numpy().astype('float32'),cmap='gray')
            plt.subplot(2,2,4)
            plt.title('union')
            plt.axis('off')
            plt.imshow(union[i].numpy().astype('float32'),cmap='gray')
            plt.show()

    inter_area = inter.sum(dim=(1,2), dtype=torch.float)
    union_area = union.sum(dim=(1,2), dtype=torch.float) + 1e-16

    # assert not (union_area == 0).any()
    return  inter_area / union_area


def iou_mask(boxes1, boxes2, xywha, mask_size=64, is_degree=True):
    r'''
    use mask method to calculate IOU between boxes1 and boxes2

    Arguments:
        boxes1: tensor or numpy, shape(N,5), 5=(x, y, w, h, angle 0~90)
        boxes2: tensor or numpy, shape(M,5), 5=(x, y, w, h, angle 0~90)
        xywha: True if xywha, False if xyxya
        mask_size: int, resolution of mask, larger -> more precise but slower
        is_degree: True if degree, False if radian

    Return:
        iou_matrix: tensor, shape(N,M), float32, 
                    ious of all possible pairs between boxes1 and boxes2
    '''
    # assert not torch.isinf(boxes2).any()
    # assert not torch.isnan(boxes2).any()
    # assert not torch.isinf(boxes1).any()
    # assert not torch.isnan(boxes1).any()
    assert xywha == True

    if not (torch.is_tensor(boxes1) and torch.is_tensor(boxes2)):
        print('Warning: bounding boxes are np.array, converting to torch.tensor')
        # convert to tensor, (batch, (x,y,w,h,a))
        boxes1 = torch.from_numpy(boxes1).float()
        boxes2 = torch.from_numpy(boxes2).float()
    
    if boxes1.dim() == 1:
        boxes1 = boxes1.unsqueeze(0)
    if boxes2.dim() == 1:
        boxes2 = boxes2.unsqueeze(0)
    assert boxes1.shape[1] == boxes2.shape[1] == 5

    num1, num2 = boxes1.shape[0], boxes2.shape[0]
    if num1 == 0 or num2 == 0:
        return torch.Tensor([]).view(num1, num2)
    boxes1 = boxes1.repeat(1,num2).view(-1,5)
    boxes2 = boxes2.repeat(num1,1)

    iou_list = iou_pairs_mask(boxes1, boxes2, xywha=True, mask_size=mask_size,
                              is_degree=is_degree)

    if num1 == 1 and num2 == 1:
        iou_matrix = iou_list.view(num1,num2) # .squeeze()
    else:
        iou_matrix = iou_list.view(num1,num2)
    return iou_matrix


from pycocotools import mask as maskUtils
def iou_rle(boxes1, boxes2, xywha, is_degree=True, **kwargs):
    r'''
    use mask method to calculate IOU between boxes1 and boxes2

    Arguments:
        boxes1: tensor or numpy, shape(N,5), 5=(x, y, w, h, angle 0~90)
        boxes2: tensor or numpy, shape(M,5), 5=(x, y, w, h, angle 0~90)
        xywha: True if xywha, False if xyxya
        is_degree: True if degree, False if radian

    Return:
        iou_matrix: tensor, shape(N,M), float32, 
                    ious of all possible pairs between boxes1 and boxes2
    '''
    assert xywha == True and is_degree == True

    if not (torch.is_tensor(boxes1) and torch.is_tensor(boxes2)):
        print('Warning: bounding boxes are np.array. converting to torch.tensor')
        # convert to tensor, (batch, (x,y,w,h,a))
        boxes1 = torch.from_numpy(boxes1).float()
        boxes2 = torch.from_numpy(boxes2).float()
    assert boxes1.device == boxes2.device
    device = boxes1.device
    boxes1, boxes2 = boxes1.cpu().clone().detach(), boxes2.cpu().clone().detach()
    if boxes1.dim() == 1:
        boxes1 = boxes1.unsqueeze(0)
    if boxes2.dim() == 1:
        boxes2 = boxes2.unsqueeze(0)
    assert boxes1.shape[1] == boxes2.shape[1] == 5
    
    size = kwargs.get('img_size', 2048)
    h,w = size if isinstance(size, tuple) else (size,size)
    if 'normalized' in kwargs and kwargs['normalized'] == True:
        # the [x,y,w,h] are between 0~1
        # assert (boxes1[:,:4] <= 1).all() and (boxes2[:,:4] <= 1).all()
        boxes1[:,0] *= w
        boxes1[:,1] *= h
        boxes1[:,2] *= w
        boxes1[:,3] *= h
        boxes2[:,0] *= w
        boxes2[:,1] *= h
        boxes2[:,2] *= w
        boxes2[:,3] *= h
    if is_degree:
        # convert to radian
        boxes1[:,4] = boxes1[:,4] * pi / 180
        boxes2[:,4] = boxes2[:,4] * pi / 180

    b1 = xywha2vertex(boxes1, is_degree=False, stack=False).tolist()
    b2 = xywha2vertex(boxes2, is_degree=False, stack=False).tolist()
    debug = 1
    
    b1 = maskUtils.frPyObjects(b1, h, w)
    b2 = maskUtils.frPyObjects(b2, h, w)
    ious = maskUtils.iou(b1, b2, [0 for _ in b2])

    return torch.from_numpy(ious).to(device=device)
