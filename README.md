# RAPiD
This repository is the official PyTorch implementation of the following paper. Our code can reproduce the training and testing results reported in the paper.

**RAPiD: Rotation-Aware People Detection in Overhead Fisheye Images** <br />
[[arXiv paper](https://arxiv.org/abs/2005.11623)] [[Project page](http://vip.bu.edu/projects/vsns/cossy/fisheye/rapid/)]

## Installation
**Requirements**:
The code should be able to work as long as you have the following packages:
- PyTorch >= 1.0. Installation instructions can be found at https://pytorch.org/get-started/locally/
- opencv-python
- [pycocotools](https://github.com/cocodataset/cocoapi) (for Windows users, please refer to [this repo](https://github.com/maycuatroi/pycocotools-window))
- tqdm
- tensorboard (optional, only for training)

An exmpale of Installation with Linux, CUDA10.1, and Conda:
```bash
conda create --name RAPiD_env python=3.7
conda activate RAPiD_env

conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
conda install -c conda-forge pycocotools
conda install tqdm opencv

# cd the_folder_to_install
git clone https://github.com/duanzhiihao/RAPiD.git
```

## Performance and pre-trained network weights
Below is the cross-validatation performance on three datasets: [Mirror Worlds](http://www2.icat.vt.edu/mirrorworlds/challenge/index.html)-[rotated bbox version](http://vip.bu.edu/projects/vsns/cossy/datasets/mw-r), [HABBOF](http://vip.bu.edu/projects/vsns/cossy/datasets/habbof/), and [CEPDOF](http://vip.bu.edu/projects/vsns/cossy/datasets/cepdof/). The links in the table refer to the pre-trained network weights that can reproduce each number.
| Resolution | MW-R | HABBOF | CEPDOF |
|:----------:|:----:|:------:|:------:|
|     608    | [96.6](https://github.com/duanzhiihao/RAPiD/releases/download/v0.1/pL1_HBCP608_Apr14_6000.ckpt) |  [97.3](https://github.com/duanzhiihao/RAPiD/releases/download/v0.1/pL1_MWCP608_Apr14_5500.ckpt)  |  [82.4](https://github.com/duanzhiihao/RAPiD/releases/download/v0.1/pL1_MWHB608_Mar11_4500.ckpt)  |
|    1024    | [96.7](https://github.com/duanzhiihao/RAPiD/releases/download/v0.1/pL1_HBCP1024_Apr14_3000.ckpt) |  [98.1](https://github.com/duanzhiihao/RAPiD/releases/download/v0.1/pL1_MWCP1024_Apr14_3000.ckpt)  |  [85.8](https://github.com/duanzhiihao/RAPiD/releases/download/v0.1/pL1_MWHB1024_Mar11_4000.ckpt)  |

## A minimum guide for testing on a single image
0. Clone the repository
1. Download the [pre-trained network weights](https://github.com/duanzhiihao/RAPiD/releases/download/v0.1/pL1_MWHB1024_Mar11_4000.ckpt), which is trained on COCO, MW-R and HABBOF, and place it under the RAPiD/weights folder.
2. Directly run `python example.py`. Alternatively, `demo.ipynb` gives an example using jupyter notebook.

<p align="center">
<img src="https://github.com/duanzhiihao/RAPiD/blob/master/readme_img/exhibition_rapid608_1024_0.3.jpg" width="500" height="500">
</p>

## Evaluation on CEPDOF
TBD

## Training on COCO
TBD

## Fine-tuning on fisheye image datasets
TBD

## TODO
- [ ] Update README

## Citation
RAPiD source code is available for non-commercial use. If you find our code and dataset useful or publish any work reporting results using this source code, please consider citing our paper
```
Z. Duan, M.O. Tezcan, H. Nakamura, P. Ishwar and J. Konrad, 
“RAPiD: Rotation-Aware People Detection in Overhead Fisheye Images”, 
in IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 
Omnidirectional Computer Vision in Research and Industry (OmniCV) Workshop, June 2020.
```
