import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt


def draw_xywha(im, x, y, w, h, angle, color=(255,0,0), linewidth=5):
    '''
    im: image numpy array, shape(h,w,3), RGB
    angle: degree
    '''
    c, s = np.cos(angle/180*np.pi), np.sin(angle/180*np.pi)
    R = np.asarray([[c, s], [-s, c]])
    pts = np.asarray([[-w/2, -h/2], [w/2, -h/2], [w/2, h/2], [-w/2, h/2]])
    rot_pts = []
    for pt in pts:
        rot_pts.append(([x, y] + pt @ R).astype(int))
    contours = np.array([rot_pts[0], rot_pts[1], rot_pts[2], rot_pts[3]])
    cv2.polylines(im, [contours], isClosed=True, color=color,
                thickness=linewidth, lineType=cv2.LINE_4)


def draw_dt_on_np(im, detections, print_dt=False, color=(255,0,0),
                  text_size=1, **kwargs):
    '''
    im: image numpy array, shape(h,w,3), RGB
    detections: rows of [x,y,w,h,a,conf], angle in degree
    '''
    line_width = kwargs.get('line_width', im.shape[0] // 300)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_bold = max(int(2*text_size), 1)
    for bb in detections:
        if len(bb) == 6:
            x,y,w,h,a,conf = bb
        else:
            x,y,w,h,a = bb[:5]
            conf = -1
        x1, y1 = x - w/2, y - h/2
        if print_dt:
            print(f'[{x} {y} {w} {h} {a}], confidence: {conf}')
        draw_xywha(im, x, y, w, h, a, color=color, linewidth=line_width)
        if kwargs.get('show_conf', True):
            cv2.putText(im, f'{conf:.2f}', (int(x1),int(y1)), font, 1*text_size,
                        (255,255,255), font_bold, cv2.LINE_AA)
        if kwargs.get('show_angle', False):
            cv2.putText(im, f'{int(a)}', (x,y), font, 1*text_size,
                        (255,255,255), font_bold, cv2.LINE_AA)


def draw_anns_on_np(im, annotations, draw_angle=False, color=(0,0,255)):
    '''
    im: image numpy array, shape(h,w,3), RGB
    annotations: list of dict, json format
    '''
    line_width = im.shape[0] // 500
    for ann in annotations:
        x, y, w, h, a = ann['bbox']
        draw_xywha(im, x, y, w, h, a, color=color, linewidth=line_width)


def flow_to_rgb(flow, plt_show=False):
    '''
    Visualizing optical flow using a RGB image

    Args:
        flow: 2xHxW tensor, flow[0,...] is horizontal motion
    '''
    assert torch.is_tensor(flow) and flow.dim() == 3 and flow.shape[0] == 2

    flow = flow.cpu().numpy()
    mag, ang = cv2.cartToPolar(flow[0, ...], flow[1, ...], angleInDegrees=True)
    hsv = np.zeros((flow.shape[1],flow.shape[2],3), dtype=np.uint8)
    hsv[..., 0] = ang / 2
    hsv[..., 1] = mag
    hsv[..., 2] = 255
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    if plt_show:
        plt.imshow(rgb)
        plt.show()
    return rgb


def tensor_to_npimg(tensor_img):
    tensor_img = tensor_img.squeeze()
    assert tensor_img.shape[0] == 3 and tensor_img.dim() == 3
    return tensor_img.permute(1,2,0).cpu().numpy()


def imshow_tensor(tensor_batch):
    batch = tensor_batch.clone().detach().cpu()
    if batch.dim() == 3:
        batch = batch.unsqueeze(0)
    for tensor_img in batch:
        np_img = tensor_to_npimg(tensor_img)
        plt.imshow(np_img)
    plt.show()


def plt_show(im):
    plt.imshow(im)
    plt.show()