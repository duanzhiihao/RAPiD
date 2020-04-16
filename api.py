# API for all the YOLOangle variants.
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from random import randint

import torch
import torchvision.transforms.functional as tvf

from utils import visualization, dataloader, utils


class Detector():
    def __init__(self, model_name='', weights_path=None, model=None, **kwargs):
        assert torch.cuda.is_available()
        if model:
            self.model = model
            return
        if model_name == 'rapid':
            from models.rapid import RAPiD
            model = RAPiD(backbone='dark53', img_norm=False, anchor_size=1024)
        else:
            raise NotImplementedError()
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'Successfully initialized model {model_name}.',
            'Total number of trainable parameters:', total_params)
        
        weights_path = weights_path or './weights/pL1_H1MW1024_Mar11_4000.ckpt'
        model.load_state_dict(torch.load(weights_path)['model'])
        print(f'Successfully loaded weights: {weights_path}')
        model.eval()
        self.model = model.cuda()
        
        # post-processing settings
        self.conf_thres = kwargs.get('conf_thres', None)
        self.input_size = kwargs.get('input_size', None)
        self.test_aug = kwargs.get('test_aug', None)
    
    def detect_one(self, **kwargs):
        '''
        Inference on a single image.

        Args:
            img_path:str or pil_img:PIL.Image
            test_aug: str, 'h','v','hv'
            input_size: int, default: 1024
            return_img: bool, default: False
            visualize: bool, default: True
        '''
        assert 'img_path' in kwargs or 'pil_img' in kwargs
        img = kwargs.get('pil_img', None) or Image.open(kwargs['img_path'])

        detections = self._predict_pil(img, **kwargs)

        if kwargs.get('return_img', False):
            np_img = np.array(img)
            visualization.draw_dt_on_np(np_img, detections, **kwargs)
            return np_img
        if kwargs.get('visualize', False):
            np_img = np.array(img)
            visualization.draw_dt_on_np(np_img, detections, **kwargs)
            plt.figure(figsize=(10,10))
            plt.imshow(np_img)
            plt.show()
        return detections
        
    def detect_imgSeq(self, img_dir, **kwargs):
        '''
        object detection in a sequence of images
        
        Args:
            img_dir: str
            test_aug: str, 'h','v','hv'
            input_size: int, default: 1024
        '''
        gt_path = kwargs['gt_path'] if 'gt_path' in kwargs else None

        ims = dataloader.Images4YOLO(img_dir, gt_path) # TODO
        dts = self._detect_iter(iter(ims), **kwargs)
        # dts = self._detect_iter_seq_nms_causal(iter(ims), **kwargs)
        return dts

    def _detect_iter(self, iterator, **kwargs):
        detection_json = []
        for _ in tqdm(range(len(iterator))):
            pil_frame, anns, img_id = next(iterator)
            detections = self._predict_pil(pil_img=pil_frame, **kwargs)

            for dt in detections:
                x, y, w, h, a, conf = [float(t) for t in dt]
                bbox = [x,y,w,h,a]
                dt_dict = {'image_id': img_id, 'bbox': bbox, 'score': conf,
                           'segmentation': []}
                detection_json.append(dt_dict)

        return detection_json
    
    def _predict_pil(self, pil_img, **kwargs):
        test_aug = kwargs.get('test_aug', self.test_aug)
        input_size = kwargs.get('input_size', self.input_size)
        conf_thres = kwargs.get('conf_thres', self.conf_thres)
        assert isinstance(pil_img, Image.Image), 'input must be a PIL.Image'

        # pad to square
        input_img, _, pad_info = utils.rect_to_square(pil_img, None, input_size, 0)
        
        input_ori = tvf.to_tensor(input_img)
        if not test_aug:
            input_ = input_ori.unsqueeze(0)
        elif test_aug == 'h':
            raise Exception('Deprecated')
            # horizontal flip
            input_hflip = input_ori.flip(2)
            input_ = torch.stack([input_ori, input_hflip], dim=0)
        elif test_aug == 'hv':
            # original, horizontal flip, vertical flip
            to_cat = [input_ori]
            to_cat.append(input_ori.flip(2)) # horizontal flip
            to_cat.append(input_ori.flip(1)) # vertical flip
            input_ = torch.stack(to_cat, dim=0)
        elif test_aug == 'rotation':
            # original, rotate 90, rotate 180, rotate270
            to_cat = [input_ori]
            to_cat.append(input_ori.rot90(-1,[1,2])) # rotate 90 degrees clockwise
            to_cat.append(input_ori.rot90(2,[1,2])) # rotate 180 degrees clockwise
            to_cat.append(input_ori.rot90(1,[1,2])) # rotate 270 degrees clockwise
            input_ = torch.stack(to_cat, dim=0)
        else:
            raise Exception('Invalid test-time augmentation')
        
        assert input_.dim() == 4
        input_ = input_.cuda()
        with torch.no_grad():
            dts = self.model(input_).cpu()

        if not test_aug:
            dts = dts.squeeze()        
            if 'enable_post' in kwargs and kwargs['enable_post'] == False:
                _, idx = torch.topk(dts[:,5], k=randint(95,105), sorted=False)
                dts = dts[idx, :]
                dts = utils.detection2original(dts, pad_info.squeeze())
                return dts
            # post-processing
            dts = dts[dts[:,5] >= conf_thres]
            if len(dts) > 1000:
                _, idx = torch.topk(dts[:,5], k=1000)
                dts = dts[idx, :]
            if kwargs.get('debug', False):
                np_img = np.array(input_img)
                visualization.draw_dt_on_np(np_img, dts)
                plt.imshow(np_img)
                plt.show()
            dts = utils.nms(dts, is_degree=True, nms_thres=0.45, img_size=input_size)
            dts = utils.detection2original(dts, pad_info.squeeze())
            if kwargs.get('debug', False):
                np_img = np.array(pil_img)
                visualization.draw_dt_on_np(np_img, dts)
                plt.imshow(np_img)
                plt.show()
            return dts

        dts_union = []
        for i, fdt in enumerate(dts):
            frames_dts = fdt[fdt[:,5] >= conf_thres]
            if len(frames_dts) > 1000:
                _, idx = torch.topk(frames_dts[:,5], k=1000)
                frames_dts = frames_dts[idx, :]
            frames_dts = utils.nms(frames_dts, is_degree=True, nms_thres=0.45)
            dts_union.append(frames_dts)
        
        if test_aug == 'rotation':    
            # frame is rotated by 90 degrees clockwise
            x,y,w,h,a,_ = dts_union[1].clone().transpose(0,1)
            dts_union[1][:,0] = y
            dts_union[1][:,1] = input_size - x
            dts_union[1][:,4] = a-90
            # frame is rotated by 180 degrees clockwise
            x,y,w,h,a,_ = dts_union[2].clone().transpose(0,1)
            dts_union[2][:,0] = input_size - x
            dts_union[2][:,1] = input_size - y
            # frame is rotated by 270 degrees clockwise (= -90 degrees clockwise)
            x,y,w,h,a,_ = dts_union[3].clone().transpose(0,1)
            dts_union[3][:,0] = input_size - y
            dts_union[3][:,1] = x
            dts_union[3][:,4] = a+90
            majority_num = 2
        elif test_aug == 'hv':
            # horizontal flipped
            dts_union[1][:,0] = input_.shape[3] - dts_union[1][:,0]
            dts_union[1][:,4] = -dts_union[1][:,0]
            # vertical flipped
            dts_union[2][:,1] = input_.shape[2] - dts_union[2][:,1]
            dts_union[2][:,4] = -dts_union[2][:,4]
            majority_num = 2
        
        # concatenate them together
        dts_union = torch.cat(dts_union, dim=0)
        dts_union = utils.nms(dts_union, is_degree=True, nms_thres=0.45,
                              majority=majority_num)
        dts_union = utils.detection2original(dts_union, pad_info.squeeze())
        return dts_union


def detect_once(model, pil_img, conf_thres, nms_thres=0.45, input_size=608):
    ori_w, ori_h = pil_img.width, pil_img.height
    input_img, _, pad_info = utils.rect_to_square(pil_img, None, input_size, 0)

    input_img = tvf.to_tensor(input_img).cuda()
    with torch.no_grad():
        dts = model(input_img[None]).cpu().squeeze()
    dts = dts[dts[:,5] >= conf_thres].cpu()
    dts = utils.nms(dts, is_degree=True, nms_thres=0.45)
    dts = utils.detection2original(dts, pad_info.squeeze())
    # np_img = np.array(pil_img)
    # api_utils.draw_dt_on_np(np_img, detections)
    # plt.imshow(np_img)
    # plt.show()
    return dts
