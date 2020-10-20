import os
import json
import random
from PIL import Image
import numpy as np
from collections import defaultdict

import torch
import torchvision.transforms.functional as tvf

from utils.utils import normalize_bbox, rect_to_square
import utils.augmentation as augUtils


class Dataset4YoloAngle(torch.utils.data.Dataset):
    """
    dataset class.
    """
    def __init__(self, img_dir, json_path, img_size=608, augmentation=True,
                 only_person=True, debug_mode=False):
        """
        dataset initialization. Annotation data are read into memory by API.

        Args:
            img_dir: str or list, imgs folder, e.g. 'someDir/COCO/train2017/'
            json_path: str or list, e.g. 'someDir/COCO/instances_train2017.json'
            img_size: int, target image size input to the YOLO, default: 608
            augmentation: bool, default: True
            only_person: bool, if true, non-person BBs are discarded. default: True
            debug: bool, if True, only one data id is selected from the dataset
        """
        self.max_labels = 50
        self.img_size = img_size
        self.enable_aug = augmentation
        self.only_person = only_person
        if only_person:
            print('Only train on person images and objects')

        self.img_ids = []
        # self.imgid2info = dict()
        self.imgid2path = dict()
        self.imgid2anns = defaultdict(list)
        self.catids = []
        if isinstance(img_dir, str):
            assert isinstance(json_path, str)
            img_dir, json_path = [img_dir], [json_path]
        assert len(img_dir) == len(json_path)
        for imdir,jspath in zip(img_dir, json_path):
            self.load_anns(imdir, jspath)

        if debug_mode:
            # self.img_ids = self.img_ids[0:1]
            self.img_ids = [428856]
            print(f"debug mode..., only train on one image: {self.img_ids[0]}")

        # transform and data augmentation
        # self.pil_aug_to_tensor = transforms.Compose([
        #     transforms.ColorJitter(brightness=0.3,contrast=0.3,saturation=1,hue=0.1),
        #     transforms.ToTensor(),
        # ])
        # self.pil_to_tensor = transforms.ToTensor()

    def load_anns(self, img_dir, json_path):
        '''
        laod json file to self.img_ids, self.imgid2anns
        '''
        self.coco = False
        print(f'Loading annotations {json_path} into memory...')
        with open(json_path, 'r') as f:
            json_data = json.load(f)
        for ann in json_data['annotations']:
            img_id = ann['image_id']
            # get width and height
            if len(ann['bbox']) == 4:
                # using COCO dataset. 4 = [x1,y1,w,h]
                self.coco = True
                # convert COCO format: x1,y1,w,h to x,y,w,h
                ann['bbox'][0] = ann['bbox'][0] + ann['bbox'][2] / 2
                ann['bbox'][1] = ann['bbox'][1] + ann['bbox'][3] / 2
                ann['bbox'].append(0)
                if ann['bbox'][2] > ann['bbox'][3]:
                    ann['bbox'][2],ann['bbox'][3] = ann['bbox'][3],ann['bbox'][2]
                    ann['bbox'][4] -= 90
            else:
                # using rotated bounding box datasets. 5 = [cx,cy,w,h,angle]
                assert len(ann['bbox']) == 5, 'Unknown bbox format' # x,y,w,h,a
            if ann['bbox'][2] == ann['bbox'][3]:
                ann['bbox'][3] += 1 # force that w < h
            ann['bbox'] = torch.Tensor(ann['bbox'])
            assert ann['bbox'][2] < ann['bbox'][3]
            if ann['bbox'][4] == 90:
                ann['bbox'][4] = -90
            assert ann['bbox'][4] >= -90 and ann['bbox'][4] < 90
            self.imgid2anns[img_id].append(ann)
        for img in json_data['images']:
            img_id = img['id']
            assert img_id not in self.imgid2path
            anns = self.imgid2anns[img_id]
            # if there is crowd gt, skip this image
            if self.coco and any(ann['iscrowd'] for ann in anns):
                continue
            # if only for person detection
            if self.only_person:
                # select the images which contain at least one person
                if not any(ann['category_id']==1 for ann in anns):
                    continue
                # and ignore all other categories
                self.imgid2anns[img_id] = [a for a in anns if a['category_id']==1]
            self.img_ids.append(img_id)
            self.imgid2path[img_id] = os.path.join(img_dir, img['file_name'])
            # self.imgid2info[img['id']] = img
        self.catids = [cat['id'] for cat in json_data['categories']]
        if self.coco:
            print('Training on perspective images; adding angle to BBs')
        else:
            assert self.only_person

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index):
        """
        One image / label pair for the given index is picked up and pre-processed.

        Args:
        index (int): data index
        """
        # laod image
        img_id = self.img_ids[index]
        # img_name = self.imgid2info[img_id]['file_name']
        # img_path = os.path.join(self.img_dir, img_name)
        img_path = self.imgid2path[img_id]
        self.coco = True if 'COCO' in img_path else False
        img = Image.open(img_path)
        ori_w, ori_h = img.width, img.height
        if img.mode == 'L':
            # print(f'Warning: image {img_id} is grayscale')
            img = np.array(img)
            img = np.repeat(np.expand_dims(img,2), 3, axis=2)
            img = Image.fromarray(img)
        # now img is a tensor with shape (3,h,w)

        # load unnormalized annotation
        annotations = self.imgid2anns[img_id]
        gt_num = len(annotations)
        # labels shape(50, 5), 5 = [x, y, w, h, angle]
        labels = torch.zeros(self.max_labels, 5)
        categories = torch.zeros(self.max_labels, dtype=torch.int64)
        li = 0
        for ann in annotations:
            if self.only_person and ann['category_id'] != 1:
                continue
            area = ann['bbox'][2]*ann['bbox'][3] / ori_w / ori_h
            if self.only_person and self.coco and area <= 0.001:
                # import matplotlib.pyplot as plt
                # plt.imshow(np.array(img))
                # plt.show()
                continue
            # assert ann['category_id'] == 1, 'only support person object'
            if li >= 50:
                print(self.only_person)
                print(categories)
                break
            labels[li,:] = ann['bbox']
            categories[li] = self.catids.index(ann['category_id'])
            li += 1
        if self.only_person:
            assert (categories == 0).all()
        gt_num = li

        # augmentation
        if self.enable_aug:
            img, labels[:gt_num] = self.augment_PIL(img, labels[:gt_num])

        # pad to square
        img, labels[:gt_num], pad_info = rect_to_square(img, labels[:gt_num],
                            self.img_size, pad_value=0, aug=self.enable_aug)
        # pad_info = torch.Tensor((ori_w, ori_h) + imtl + imwh)

        img = tvf.to_tensor(img)
        if self.enable_aug:
            if np.random.rand() > 0.5:
                img = augUtils.add_gaussian(img, max_var=0.03)
            blur = [augUtils.random_avg_filter, augUtils.max_filter,
                    augUtils.random_gaussian_filter]
            if not self.coco and np.random.rand() > 0.8:
                blur_func = random.choice(blur)
                img = blur_func(img)
            if np.random.rand() > 0.5:
                img = augUtils.add_saltpepper(img, max_p=0.04)

        labels[:gt_num] = normalize_bbox(labels[:gt_num], self.img_size, self.img_size)

        # x,y,w,h: 0~1, angle: -90~90 degrees
        assert img.dim() == 3 and img.shape[0] == 3 and img.shape[1] == img.shape[2]
        assert (labels[:,2] <= labels[:,3]).all(), f'{labels[labels[:,2]>labels[:,3]]}'
        return img, labels, categories, str(img_id), pad_info

    def augment_PIL(self, img, labels):
        if np.random.rand() > 0.4:
            img = tvf.adjust_brightness(img, uniform(0.3,1.5))
        if np.random.rand() > 0.7:
            factor = 2 ** uniform(-1, 1)
            img = tvf.adjust_contrast(img, factor) # 0.5 ~ 2
        if np.random.rand() > 0.7:
            img = tvf.adjust_hue(img, uniform(-0.1,0.1))
        if np.random.rand() > 0.6:
            factor = uniform(0,2)
            if factor > 1:
                factor = 1 + uniform(0, 2)
            img = tvf.adjust_saturation(img, factor) # 0 ~ 3
        if np.random.rand() > 0.5:
            img = tvf.adjust_gamma(img, uniform(0.5, 3))
        # horizontal flip
        if np.random.rand() > 0.5:
            img, labels = augUtils.hflip(img, labels)
        # vertical flip
        if np.random.rand() > 0.5:
            img, labels = augUtils.vflip(img, labels)
        # # random rotation
        rand_degree = np.random.rand() * 360
        if self.coco:
            img, labels = augUtils.rotate(img, rand_degree, labels, expand=True)
        else:
            img, labels = augUtils.rotate(img, rand_degree, labels, expand=False)
        return img, labels



def uniform(a, b):
    return a + np.random.rand() * (b-a)