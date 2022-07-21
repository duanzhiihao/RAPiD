# This is the script for training models
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import argparse
import random
import numpy as np
import cv2
import torch
import torch.cuda.amp as amp
from torch.utils.data import DataLoader

from models.rapid import RAPiD
from datasets import Dataset4YoloAngle
from utils import timer
from utils.utils import query_yes_no
from utils.visualization import draw_dt_on_np
from utils.eval_tools import CustomRotBBoxEval
import api


def parse_args():
    parser = argparse.ArgumentParser()
    # dataset
    parser.add_argument('--train_data', type=str,   default='coco')
    parser.add_argument('--high_res',   action=argparse.BooleanOptionalAction, default=False)
    # pre-trained weights
    parser.add_argument('--pretrain',   type=str,   default=None)
    # training
    parser.add_argument('--batch_size', type=int,   default=8)
    parser.add_argument('--iterations', type=int,   default=100_000)
    parser.add_argument('--amp',        action=argparse.BooleanOptionalAction, default=True)
    # optimizer
    parser.add_argument('--lr',         type=float, default=0.001)
    parser.add_argument('--momentum',   type=float, default=0.9)
    # device
    parser.add_argument('--device',     type=str,   default='cuda:0')
    parser.add_argument('--workers',    type=int,   default=0)
    # logging
    parser.add_argument('--eval_interval',       type=int, default=1000)
    parser.add_argument('--print_interval',      type=int, default=10)
    parser.add_argument('--checkpoint_interval', type=int, default=1000)
    return parser.parse_args()


class Evaluator():
    def __init__(self, img_dir, json_path):
        self.img_dir = img_dir
        self.evaluator = CustomRotBBoxEval(json_path)

    @torch.no_grad()
    def evaluate(self, model, target_size):
        model.eval()
        print(f'Evaluting on images in {self.img_dir} ...')
        model_eval = api.Detector(conf_thres=0.01, model=model)
        dts = model_eval.detect_imgSeq(self.img_dir, input_size=target_size)
        msg = self.evaluator.evaluate_dtList(dts, metric='AP')
        print(msg)
        model.train()


def get_dataset(cfg):
    coco_root = Path('../../datasets/coco')
    data_root = Path('../../datasets/fisheye')
    ann_root  = Path('../../datasets/fisheye/annotations')
    known_splits = {
        'coco':     dict(img_dir=coco_root/'train2017', json_path=coco_root/'annotations/instances_train2017.json'),
        'tiny-val': dict(img_dir='images/tiny_val/one', json_path='images/tiny_val/one.json'),
        'mw-r':     dict(img_dir=data_root / 'MW-R',    json_path=ann_root / 'MW-R.json'),
        'meeting1': dict(img_dir=data_root / 'HABBOF/Meeting1',     json_path=ann_root / 'HABBOF/Meeting1.json'),
        'meeting2': dict(img_dir=data_root / 'HABBOF/Meeting2',     json_path=ann_root / 'HABBOF/Meeting2.json'),
        'lab1':     dict(img_dir=data_root / 'HABBOF/Lab1',         json_path=ann_root / 'HABBOF/Lab1.json'),
        'lab2':     dict(img_dir=data_root / 'HABBOF/Lab2',         json_path=ann_root / 'HABBOF/Lab2.json'),
        'lunch1':   dict(img_dir=data_root / 'CEPDOF/Lunch1',       json_path=ann_root / 'CEPDOF/Lunch1.json'),
        'lunch2':   dict(img_dir=data_root / 'CEPDOF/Lunch2',       json_path=ann_root / 'CEPDOF/Lunch2.json'),
        'lunch3':   dict(img_dir=data_root / 'CEPDOF/Lunch3',       json_path=ann_root / 'CEPDOF/Lunch3.json'),
        'edgecase': dict(img_dir=data_root / 'CEPDOF/Edge_cases',   json_path=ann_root / 'CEPDOF/Edge_cases.json'),
        'high-act': dict(img_dir=data_root / 'CEPDOF/High_activity',json_path=ann_root / 'CEPDOF/High_activity.json'),
        'irill':    dict(img_dir=data_root / 'CEPDOF/IRill',        json_path=ann_root / 'CEPDOF/IRill.json'),
    }
    if cfg.train_data == 'coco':
        train_data = ['coco']
        val_data   = 'tiny-val'
    elif cfg.train_data == 'hbcp': # HABBOF and CEPDOF
        train_data = ['meeting1', 'meeting2', 'lab2', 'lunch1', 'lunch2', 'lunch3', 'edgecase', 'high-act', 'irill']
        val_data   = 'lab1'
    elif cfg.train_data == 'hbmw': # HABBOF and MW-R
        train_data = ['meeting1', 'meeting2', 'lab2', 'mw-r']
        val_data   = 'lab1'
    elif cfg.train_data == 'cpmw': # CEPDOF and MW-R
        train_data = ['lunch1', 'lunch2', 'edgecase', 'high-act', 'irill', 'mw-r']
        val_data   = 'lunch3'
    else:
        raise ValueError(f'Unknown dataset name {cfg.train_data}')

    train_img_dir = [known_splits[name]['img_dir']   for name in train_data]
    train_json    = [known_splits[name]['json_path'] for name in train_data]
    trainset = Dataset4YoloAngle(train_img_dir, train_json, img_size=cfg.initial_size, only_person=True)
    evaluator = Evaluator(known_splits[val_data]['img_dir'], known_splits[val_data]['json_path'])
    return trainset, evaluator


def make_infinite_trainloader(dataset, cfg):
    trainloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, 
                            num_workers=cfg.workers, pin_memory=True, drop_last=False)
    while True:
        yield from trainloader


def random_resizing_(trainset, cfg):
    if cfg.high_res:
        imgsize = random.randint(16, 34) * 32
    else:
        low = 10 if (cfg.train_data == 'coco') else 16
        imgsize = random.randint(low, 21) * 32
    trainset.img_size = imgsize
    trainloader = make_infinite_trainloader(trainset, cfg)
    return trainloader


def get_optimizer(model, cfg):
    parameters = model.parameters()
    optimizer = torch.optim.SGD(parameters, lr=cfg.lr, momentum=cfg.momentum)
    return optimizer


def adjust_lr_(optimizer, i, cfg):
    warm_up = 500
    lr_min = 1e-3
    if i < warm_up:
        factor = ((i+1) / warm_up) ** 2
    else:
        factor = lr_min + 0.5 * (1 - lr_min) * (1 + np.cos(i * np.pi / cfg.iterations))
    # update learning rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = cfg.lr * factor


def save_checkpoint(save_path, model, optimizer=None, iter_i=None):
    checkpoint = {
        'iter': iter_i,
        'model': model.state_dict(),
    }
    if optimizer is not None:
        checkpoint['optimizer'] = optimizer.state_dict()
    torch.save(checkpoint, save_path)


def check_directory(log_dir):
    if log_dir.is_dir():
        flag = query_yes_no(f'{log_dir} already exists. Still log into this directory?')
        if not flag:
            exit()
    else:
        log_dir.mkdir(parents=True)


def pbar_log(pbar, optimizer, iter_i, cfg, loss, img_size, loss_str):
    current_lr = optimizer.param_groups[0]['lr']
    max_cuda = torch.cuda.max_memory_allocated(0) / 1e9
    msg =  f'Iter: {iter_i}/{cfg.iterations} || '
    msg += f'GPU_mem={max_cuda}G || '
    msg += f'lr={current_lr} || '
    msg += f'loss={loss.item():.3f} || '
    msg += f'img_size={img_size} || '
    pbar.set_description(msg)
    torch.cuda.reset_peak_memory_stats(0)

    if (cfg.print_interval > 0) and (iter_i % cfg.print_interval == 0):
        print(loss_str)


def detect_and_save(model, target_size, log_dir):
    model.eval()
    eval_img_paths = Path('images').glob('*.jpg')
    for img_path in eval_img_paths:
        eval_img = Image.open(img_path)
        dts = api.detect_once(model, eval_img, conf_thres=0.1, input_size=target_size)
        np_img = np.array(eval_img)
        draw_dt_on_np(np_img, dts)
        cv2.imwrite(str(log_dir / 'detects.png'), np_img)
        np_img = cv2.resize(np_img, (416,416))
        # logger.add_image(img_path, np_img, iter_i, dataformats='HWC')
    model.train()


def main():
    cfg = parse_args()

    # -------------------------- settings ---------------------------
    assert torch.cuda.is_available() # Currently do not support CPU training
    device = torch.device(cfg.device)
    # ---- training data setting ----
    cfg.target_size  = 1024 if cfg.high_res else 608
    cfg.initial_size = 1088 if cfg.high_res else 672
    cfg.multiscale_interval = 10
    # ---- minibatch setting ----
    cfg.subdivision = 128 // cfg.batch_size
    print(f'effective batch size = {cfg.batch_size} * {cfg.subdivision}')
    # ---- logging setting ----
    log_dir = Path(f'runs/rapid-{cfg.train_data}-{timer.today()}')
    check_directory(log_dir)
    cfg.img_interval = 100

    # -------------------------- dataset ---------------------------
    print('initialing dataloader...')
    trainset, evaluator = get_dataset(cfg)
    trainloader = make_infinite_trainloader(trainset, cfg)

    # -------------------------- model ---------------------------
    model = RAPiD(loss_angle='period_L1')
    model = model.to(device=device)
    # load pretain
    if cfg.pretrain is not None:
        print(f'loading ckpt from {cfg.pretrain} ...')
        checkpoint = torch.load(cfg.pretrain)
        model.load_state_dict(checkpoint['model'])

    # -------------------------- optimizer ---------------------------
    optimizer = get_optimizer(model, cfg)
    scaler = amp.GradScaler(enabled=cfg.amp) # Automatic Mixed Precision

    # start training loop
    pbar = tqdm(range(cfg.iterations)) # progress bar
    for iter_i in pbar:
        # evaluation
        if (iter_i % cfg.eval_interval == 0) and (cfg.train_data != 'coco' or iter_i > 0):
            evaluator.evaluate(model, target_size=cfg.target_size)

        # learning rate schedule
        adjust_lr_(optimizer, iter_i, cfg)

        # gradient accumulation loop
        optimizer.zero_grad()
        for _ in range(cfg.subdivision):
            imgs, targets, cats, _, _ = next(trainloader) # load a batch
            imgs = imgs.to(device=device)
            with amp.autocast(enabled=cfg.amp):
                loss = model(imgs, targets, labels_cats=cats)
                loss = loss / float(cfg.subdivision)
            scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # logging
        pbar_log(pbar, optimizer, iter_i, cfg, loss, trainset.img_size, model.loss_str)

        # random resizing
        if (iter_i > 0) and (iter_i % cfg.multiscale_interval == 0):
            trainloader = random_resizing_(trainset, cfg)

        # save checkpoint
        if (iter_i > 0) and (iter_i % cfg.checkpoint_interval == 0):
            save_checkpoint(log_dir / 'last.pt', model, optimizer, iter_i)

        # save detection
        if (iter_i > 0) and (iter_i % cfg.img_interval == 0):
            detect_and_save(model, cfg.target_size, log_dir)

    print('Training end!')


if __name__ == '__main__':
    main()
