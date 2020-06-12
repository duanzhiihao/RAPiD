import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torchvision.transforms.functional as tvf

from utils import visualization, dataloader, utils


class Detector():
    '''
    Wrapper of image object detectors.

    Args:
        model_name: str, currently only support 'rapid'
        weights_path: str, path to the pre-trained network weights
        model: torch.nn.Module, used only during training
        conf_thres: float, confidence threshold
        input_size: int, input resolution
    '''
    def __init__(self, model_name='', weights_path=None, model=None, **kwargs):
        assert torch.cuda.is_available()
        if model:
            self.model = model
            return
        if model_name == 'rapid':
            from models.rapid import RAPiD
            model = RAPiD(backbone='dark53')
        else:
            raise NotImplementedError()
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'Successfully initialized model {model_name}.',
            'Total number of trainable parameters:', total_params)
        
        model.load_state_dict(torch.load(weights_path)['model'])
        print(f'Successfully loaded weights: {weights_path}')
        model.eval()
        self.model = model.cuda()
        
        # post-processing settings
        self.conf_thres = kwargs.get('conf_thres', None)
        self.input_size = kwargs.get('input_size', None)
    
    def detect_one(self, **kwargs):
        '''
        Inference on a single image.

        Args:
            img_path: str or pil_img: PIL.Image

            input_size: int, input resolution
            conf_thres: float, confidence threshold

            return_img: bool, if True, return am image with bbox visualizattion. \
                default: False
            visualize: bool, if True, plt.show the image with bbox visualization. \
                default: False
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
        Run on a sequence of images in a folder.
        
        Args:
            img_dir: str
            input_size: int, input resolution
            conf_thres: float, confidence threshold
        '''
        gt_path = kwargs['gt_path'] if 'gt_path' in kwargs else None

        ims = dataloader.Images4Detector(img_dir, gt_path) # TODO
        dts = self._detect_iter(iter(ims), **kwargs)
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
        '''
        Args:
            pil_img: PIL.Image.Image
            input_size: int, input resolution
            conf_thres: float, confidence threshold
        '''
        input_size = kwargs.get('input_size', self.input_size)
        conf_thres = kwargs.get('conf_thres', self.conf_thres)
        assert isinstance(pil_img, Image.Image), 'input must be a PIL.Image'
        assert input_size is not None, 'Please specify the input resolution'
        assert conf_thres is not None, 'Please specify the confidence threshold'

        # pad to square
        input_img, _, pad_info = utils.rect_to_square(pil_img, None, input_size, 0)
        
        input_ori = tvf.to_tensor(input_img)
        input_ = input_ori.unsqueeze(0)
        
        assert input_.dim() == 4
        input_ = input_.cuda()
        with torch.no_grad():
            dts = self.model(input_).cpu()

        dts = dts.squeeze()
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


def detect_once(model, pil_img, conf_thres, nms_thres=0.45, input_size=608):
    '''
    Run the model on the pil_img and return the detections.
    '''
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
