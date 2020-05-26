# some utils codes for api.py, e.g., image loader, video loader
import os
import json
from collections import defaultdict

import torch
import numpy as np
import cv2
from PIL import Image


class Video4Detector():
    def __init__(self, video_path):
        self.video_path = video_path
    
    def __len__(self):
        return self.total_frame_num

    def __iter__(self):
        # load video
        video = cv2.VideoCapture(self.video_path)
        # attributes
        self.total_frame_num = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.fps = int(video.get(cv2.CAP_PROP_FPS))
        self.video = video
        self.current_frame = 0
        return self

    def __next__(self):
        flag, frame = self.video.read()
        self.current_frame += 1

        assert flag == True and self.video.isOpened()
        assert self.current_frame == int(self.video.get(cv2.CAP_PROP_POS_FRAMES))

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        return frame, None, None

    def close(self):
        self.video.release()


class Images4Detector():
    def __init__(self, images_dir, gt_json=None, img_type='PIL'):
        '''
        img_type: str, one of 'PIL', 'cv2', 'plt'
        '''
        # images
        def is_img(s):
            return s.endswith('.jpg') or s.endswith('.png')
        self.img_names = [s for s in os.listdir(images_dir) if is_img(s)]
        self.img_names.sort()
        self.img_dir = images_dir
        if img_type == 'PIL':
            print('Using PIL.Image format')
            self.imread = Image.open
        elif img_type == 'cv2':
            print('Using cv2 image format, i.e., BGR')
            self.imread = cv2.imread
        elif img_type == 'plt':
            print('Using plt image format, i.e., standard RGB')
            import matplotlib.pyplot as plt
            self.imread = plt.imread
        # ground truths
        if gt_json:
            self.load_gt(gt_json)
        else:
            self.imgid2anns = None
        # attributes
        self.total_frame_num = len(self.img_names)
        first = Image.open(os.path.join(self.img_dir, self.img_names[0]))
        self.frame_h = first.height
        self.frame_w = first.width
    
    def load_gt(self, gt_json):
        with open(gt_json, 'r') as f:
            json_data = json.load(f)
        imgid2anns = defaultdict(list)
        for ann in json_data['annotations']:
            img_id = ann['image_id']
            ann['bbox'] = torch.Tensor(ann['bbox'])
            imgid2anns[img_id].append(ann)
        self.imgid2anns = imgid2anns

    def __len__(self):
        return self.total_frame_num
    
    def __iter__(self):
        self.i = -1
        return self
    
    def __next__(self):
        self.i += 1
        img_name = self.img_names[self.i]
        # load frame
        img_path = os.path.join(self.img_dir, img_name)
        frame = self.imread(img_path)
        # assert frame.width == self.frame_w and frame.height == self.frame_h
        # load ground truth
        image_id = img_name[:-4]
        if self.imgid2anns:
            anns = self.imgid2anns[image_id]
        else:
            anns = None
        return frame, anns, image_id
