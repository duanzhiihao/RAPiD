# This is the script for training models
from pathlib import Path
from PIL import Image
import argparse
import random
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader

from models.rapid import RAPiD
from datasets import Dataset4YoloAngle
from utils import MWtools, visualization
import api


def parse_args():
    parser = argparse.ArgumentParser()
    # dataset
    parser.add_argument('--train_data', type=str, default='coco')
    parser.add_argument('--high_resolution', action='store_true')
    # training
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr',         type=float, default=0.001)
    parser.add_argument('--workers',    type=int, default=0)
    # pre-train
    parser.add_argument('--checkpoint', type=str, default='')
    # logging
    parser.add_argument('--eval_interval',       type=int, default=1000)
    parser.add_argument('--img_interval',        type=int, default=500)
    parser.add_argument('--print_interval',      type=int, default=1)
    parser.add_argument('--checkpoint_interval', type=int, default=2000)
    return parser.parse_args()


def burnin_schedule(i):
    burn_in = 500
    if i < burn_in:
        factor = (i / burn_in) ** 2
    elif i < 30000:
        factor = 1.0
    elif i < 40000:
        factor = 0.5
    elif i < 100000:
        factor = 0.2
    elif i < 300000:
        factor = 0.1
    else:
        factor = 0.01
    return factor

# Learning rate setup
def burnin_schedule(i):
    burn_in = 500
    if i < burn_in:
        factor = (i / burn_in) ** 2
    elif i < 10000:
        factor = 1.0
    elif i < 20000:
        factor = 0.3
    else:
        factor = 0.1
    return factor

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
    if cfg.dataset == 'coco':
        train_data = ['coco']
        val_data   = 'tiny-val'
    elif cfg.dataset == 'hbcp': # HABBOF and CEPDOF
        train_data = ['meeting1', 'meeting2', 'lab2', 'lunch1', 'lunch2', 'lunch3', 'edgecase', 'high-act', 'irill']
        val_data   = 'lab1'
    elif cfg.dataset == 'hbmw': # HABBOF and MW-R
        train_data = ['meeting1', 'meeting2', 'lab2', 'mw-r']
        val_data   = 'lab1'
    elif cfg.dataset == 'cpmw': # CEPDOF and MW-R
        train_data = ['lunch1', 'lunch2', 'edgecase', 'high-act', 'irill', 'mw-r']
        val_data   = 'lunch3'
    else:
        raise ValueError(f'Unknown dataset name {cfg.dataset}')

    train_img_dir = [data['img_dir']   for data in train_data]
    train_json    = [data['json_path'] for data in train_data]
    trainset = Dataset4YoloAngle(train_img_dir, train_json, img_size=cfg.initial_size, only_person=True)
    dataloader = DataLoader(trainset, batch_size=cfg.batch_size, shuffle=True, 
                            num_workers=cfg.workers, pin_memory=True, drop_last=False)
    return trainset


def get_optimizer(model, cfg):
    if cfg.dataset not in cfg.checkpoint:
        start_iter = -1
    else:
        optimizer.load_state_dict(state['optimizer_state_dict'])
        print(f'begin from iteration: {start_iter}')
    # optimizer setup
    params = []
    # set weight decay only on conv.weight
    for key, value in model.named_parameters():
        if 'conv.weight' in key:
            params += [{'params':value, 'weight_decay':decay_SGD}]
        else:
            params += [{'params':value, 'weight_decay':0.0}]

    lr_SGD = 0.0001 / cfg.batch_size / cfg.subdivision
    if cfg.dataset == 'COCO':
        optimizer = torch.optim.SGD(params, lr=lr_SGD, weight_decay=decay_SGD)
    elif cfg.dataset in {'MW', 'HBCP', 'HBMW', 'CPMW'}:
        assert cfg.checkpoint is not None
        optimizer = torch.optim.SGD(params, lr=lr_SGD)
    else:
        raise NotImplementedError()


def main():
    cfg = parse_args()
    assert torch.cuda.is_available() # Currently do not support CPU training

    # -------------------------- settings ---------------------------
    cfg.target_size  = 1024 if cfg.high_resolution else 608
    cfg.initial_size = 1088 if cfg.high_resolution else 672
    # job_name = f'{cfg.model}_{cfg.dataset}{cfg.target_size}'
    # dataloader setting
    num_cpu = 0 if cfg.batch_size == 1 else 4
    subdivision = 128 // cfg.batch_size
    multiscale = True
    multiscale_interval = 10
    # SGD optimizer
    decay_SGD = 0.0005 * cfg.batch_size * subdivision
    print(f'effective batch size = {cfg.batch_size} * {subdivision}')

    # -------------------------- dataset ---------------------------
    print('initialing dataloader...')
    trainset = get_dataset(cfg)

    # -------------------------- model ---------------------------
    model = RAPiD(loss_angle='period_L1')
    model = model.cuda()

    start_iter = -1
    if cfg.checkpoint:
        print("loading ckpt...", cfg.checkpoint)
        weights_path = os.path.join('./weights/', cfg.checkpoint)
        state = torch.load(weights_path)
        model.load_state_dict(state['model'])
        start_iter = state['iter']

    val_set = MWtools.MWeval(val_json, iou_method='rle')
    eval_img_names = os.listdir('./images/')
    eval_img_paths = [os.path.join('./images/',s) for s in eval_img_names if s.endswith('.jpg')]
    # logger = SummaryWriter(f'./logs/{job_name}')


    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, burnin_schedule, last_epoch=start_iter)

    # start training loop
    today = timer.today()
    start_time = timer.tic()
    for iter_i in range(start_iter, 500000):
        # evaluation
        if iter_i % cfg.eval_interval == 0 and (cfg.dataset != 'COCO' or iter_i > 0):
            with timer.contexttimer() as t0:
                model.eval()
                model_eval = api.Detector(conf_thres=0.005, model=model)
                dts = model_eval.detect_imgSeq(val_img_dir, input_size=target_size)
                str_0 = val_set.evaluate_dtList(dts, metric='AP')
            s = f'\nCurrent time: [ {timer.now()} ], iteration: [ {iter_i} ]\n\n'
            s += str_0 + '\n\n'
            s += f'Validation elapsed time: [ {t0.time_str} ]'
            print(s)
            logger.add_text('Validation summary', s, iter_i)
            logger.add_scalar('Validation AP[IoU=0.5]', val_set._getAP(0.5), iter_i)
            logger.add_scalar('Validation AP[IoU=0.75]', val_set._getAP(0.75), iter_i)
            logger.add_scalar('Validation AP[IoU=0.5:0.95]', val_set._getAP(), iter_i)
            model.train()

        # subdivision loop
        optimizer.zero_grad()
        for inner_iter_i in range(subdivision):
            try:
                imgs, targets, cats, _, _ = next(dataiterator)  # load a batch
            except StopIteration:
                dataiterator = iter(dataloader)
                imgs, targets, cats, _, _ = next(dataiterator)  # load a batch
            # visualization.imshow_tensor(imgs)
            imgs = imgs.cuda()
            loss = model(imgs, targets, labels_cats=cats)
            loss.backward()
        optimizer.step()
        scheduler.step()

        # logging
        if iter_i % cfg.print_interval == 0:
            sec_used = timer.tic() - start_time
            time_used = timer.sec2str(sec_used)
            avg_iter = timer.sec2str(sec_used/(iter_i+1-start_iter))
            avg_epoch = avg_iter / batch_size / subdivision * 118287
            print(f'\nTotal time: {time_used}, iter: {avg_iter}, epoch: {avg_epoch}')
            current_lr = scheduler.get_last_lr()[0] * batch_size * subdivision
            print(f'[Iteration {iter_i}] [learning rate {current_lr:.3g}]',
                  f'[Total loss {loss:.2f}] [img size {dataset.img_size}]')
            print(model.loss_str)
            max_cuda = torch.cuda.max_memory_allocated(0) / 1024 / 1024 / 1024
            print(f'Max GPU memory usage: {max_cuda} GigaBytes')
            torch.cuda.reset_peak_memory_stats(0)

        # random resizing
        if multiscale and iter_i > 0 and (iter_i % multiscale_interval == 0):
            if cfg.high_resolution:
                imgsize = random.randint(16, 34) * 32
            else:
                low = 10 if cfg.dataset == 'COCO' else 16
                imgsize = random.randint(low, 21) * 32
            dataset.img_size = imgsize
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                num_workers=num_cpu, pin_memory=True, drop_last=False)
            dataiterator = iter(dataloader)

        # save checkpoint
        if iter_i > 0 and (iter_i % cfg.checkpoint_interval == 0):
            state_dict = {
                'iter': iter_i,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            save_path = os.path.join('./weights', f'{job_name}_{today}_{iter_i}.ckpt')
            torch.save(state_dict, save_path)

        # save detection
        if iter_i > 0 and iter_i % cfg.img_interval == 0:
            for img_path in eval_img_paths:
                eval_img = Image.open(img_path)
                dts = api.detect_once(model, eval_img, conf_thres=0.1, input_size=target_size)
                np_img = np.array(eval_img)
                visualization.draw_dt_on_np(np_img, dts)
                np_img = cv2.resize(np_img, (416,416))
                # cv2.imwrite(f'./results/eval_imgs/{job_name}_{today}_{iter_i}.jpg', np_img)
                logger.add_image(img_path, np_img, iter_i, dataformats='HWC')

            model.train()
    
    debug = 1


if __name__ == '__main__':
    main()

