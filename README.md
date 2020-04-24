# RAPiD
This repository is the official PyTorch implementation of the following paper. Our code can reproduce the **training** and **testing** results reported in the paper.

**RAPiD: Rotation-Aware People Detection in Overhead Fisheye Images** <br />
[Paper: publishing soon] [[Project page](http://vip.bu.edu/projects/vsns/cossy/fisheye/rapid/)]

## Installation
**Requirements**:
The code should be able to work as long as you have the following packages:
- PyTorch >= 1.0. Installation instructions can be found at https://pytorch.org/get-started/locally/
- opencv-python
- [pycocotools](https://github.com/cocodataset/cocoapi) (for Windows users, please refer to [this repo](https://github.com/maycuatroi/pycocotools-window))
- tqdm
- tensorboard (only for training)

**Installation** with Linux, CUDA10.1, and Conda:
```bash
conda create --name RAPiD_env python=3.7
conda activate RAPiD_env

conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
conda install -c conda-forge pycocotools
conda install tqdm opencv

git clone https://github.com/duanzhiihao/RAPiD.git
```

## Testing on one image
`demo.ipynb` gives an example of testing on a single image and visualizing the results.

<p align="center">
<img src="https://github.com/duanzhiihao/RAPiD/blob/master/readme_img/exhibition_rapid608_1024_0.3.jpg" width="500" height="500">
</p>

## Evaluation on CEPDOF
TBD

## Training on COCO
TBD

## Fine-tuning on CEPDOF
TBD

## TODO
- [ ] Upload network weights
- [ ] Update README

## Citation
RAPiD source code is available for non-commercial use. If you find our code or dataset useful, or publish any work reporting results using this source code, please consider citing our paper
```
Z. Duan, M.O. Tezcan, H. Nakamura, P. Ishwar and J. Konrad, 
“RAPiD: Rotation-Aware People Detection in Overhead Fisheye Images”, 
in IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 
Omnidirectional Computer Vision in Research and Industry (OmniCV) Workshop, June 2020.
```
