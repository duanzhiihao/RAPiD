# RAPiD
This repository is the official PyTorch implementation of the following paper:

**RAPiD: Rotation-Aware People Detection in Overhead Fisheye Images** <br />
[Paper: publishing soon] [[Project page](http://vip.bu.edu/projects/vsns/cossy/fisheye/rapid/)]

Our code can reproduce the **training** and **testing** results reported in the paper.

## Requirements
In short, the code should work under
- PyTorch >= 1.0. Installation instructions can be found at https://pytorch.org/get-started/locally/
- opencv-python
- [pycocotools](https://github.com/cocodataset/cocoapi) (for Windows users, please refer to [this repo](https://github.com/maycuatroi/pycocotools-window))
- tqdm
- tensorboard (only for training)

The detailed environment information is listed in `requirements.txt`.

## Testing on one image
TBD

## Evaluation on CEPDOF
TBD

## Training on COCO
TBD

## Fine-tuning on CEPDOF
TBD

## TODO
- [ ] Update README

## Citation
If you find our code or dataset useful, please consider citing our paper:
```
Z. Duan, M.O. Tezcan, H. Nakamura, P. Ishwar and J. Konrad, 
“RAPiD: Rotation-Aware People Detection in Overhead Fisheye Images”, 
in IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 
Omnidirectional Computer Vision in Research and Industry (OmniCV) Workshop, June 2020.
```
