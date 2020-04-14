# This is the main training file we are using
import os
import argparse
import random
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np
import cv2

from datasets import Dataset4YoloAngle
from utils import MWtools, timer, visualization
import api


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--SCC', action='store_true')
    parser.add_argument('--model', type=str, default='ex_pL1')
    parser.add_argument('--backbone', type=str, default='dark53')
    parser.add_argument('--dataset', type=str, default='THH1MW')
    parser.add_argument('--batch_size', type=int, default=1)

    parser.add_argument('--high_resolution', action='store_true')
    # parser.add_argument('--img_norm', action='store_true')

    parser.add_argument('--checkpoint', type=str, default='exact_pL1_dark53_H1MW1024_Mar11_4000.ckpt')
    # parser.add_argument('--load_optim', action='store_true')
    # parser.add_argument('--optim', type=str, default='SGDMR')

    parser.add_argument('--eval_interval', type=int, default=1000)
    parser.add_argument('--img_interval', type=int, default=500)
    parser.add_argument('--print_interval', type=int, default=1)
    parser.add_argument('--checkpoint_interval', type=int, default=2000)
    
    parser.add_argument('--debug', action='store_true') # default=True)
    parser.add_argument('--cuda', type=bool, default=True) # default=True)
    parser.add_argument('--skip_first_eval', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    assert torch.cuda.is_available()
    # -------------------------- settings ---------------------------
    # assert not args.adversarial
    target_size = 1024 if args.high_resolution else 608
    initial_size = 1088 if args.high_resolution else 672
    job_name = f'{args.model}_{args.backbone}_{args.dataset}{target_size}'
    # dataloader setting
    only_person = False if args.model == 'Hitachi80' else True
    print('Only train on person images and object:', only_person)
    batch_size = args.batch_size
    num_cpu = 0 if batch_size == 1 else 4
    subdivision = 128 // batch_size
    enable_aug = True
    multiscale = True # if (args.model != 'alpha_fc' and enable_aug) else False
    multiscale_interval = 10
    # SGD optimizer
    decay_SGD = 0.0005 * batch_size * subdivision
    print(f'effective batch size = {batch_size} * {subdivision}')
    # dataset setting
    print('initialing dataloader...')
    if args.dataset == 'COCO':
        train_img_dir = '../../../COCO/train2017'
        train_json = '../../../COCO/annotations/instances_train2017.json'
        val_img_dir = '../../../COSSY/valJan/'
        val_json = '../../../COSSY/annotations/valJan.json'
        lr_SGD = 0.001 / batch_size / subdivision
        # Learning rate setup
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
    elif args.dataset == 'THEODORE':
        train_img_dir = '../../../THEODORE/images'
        train_json = '../../../THEODORE/annotations/person_bbox.json'
        val_img_dir = '../../../COSSY/Lab1/'
        val_json = '../../../COSSY/annotations/Lab1.json'
        lr_SGD = 0.0001 / batch_size / subdivision
        # Learning rate setup
        def burnin_schedule(i):
            burn_in = 500
            if i < burn_in:
                factor = (i / burn_in) ** 2
            elif i < 20000:
                factor = 1.0
            elif i < 40000:
                factor = 0.3
            else:
                factor = 0.1
            return factor
    elif args.dataset == 'MW':
        train_img_dir = '../../../MW18Mar/whole'
        train_json = '../../../MW18Mar/annotations/no19_nosmall.json'
        val_img_dir = '../../../COSSY/valJan/'
        val_json = '../../../COSSY/annotations/valJan.json'
        lr_SGD = 0.0001 / batch_size / subdivision
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
    elif args.dataset == 'H1H2':
        videos = ['Meeting1', 'Meeting2', 'Lab2',
                  'Lunch1', 'Lunch2', 'Lunch3', 'Edge_cases', 'IRill', 'Activity']
        # if args.high_resolution:
        #     videos += ['All_off', 'IRfilter', 'IRill']
        train_img_dir = [f'../../../COSSY/{s}/' for s in videos]
        train_json = [f'../../../COSSY/annotations/{s}.json' for s in videos]
        val_img_dir = '../../../COSSY/Lab1/'
        val_json = '../../../COSSY/annotations/Lab1.json'
        lr_SGD = 0.0001 / batch_size / subdivision
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
    elif args.dataset == 'H1MW':
        videos = ['Meeting1', 'Meeting2', 'Lab2', 'MW']
        train_img_dir = [f'../../../COSSY/{s}/' for s in videos]
        train_json = [f'../../../COSSY/annotations/{s}.json' for s in videos]
        val_img_dir = '../../../COSSY/Lab1/'
        val_json = '../../../COSSY/annotations/Lab1.json'
        lr_SGD = 0.0001 / batch_size / subdivision
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
    elif args.dataset == 'H2MW':
        videos = ['Lunch1', 'Lunch2', 'Edge_cases', 'IRill', 'Activity',
                  'MW']
        # if args.high_resolution:
        #     videos += ['All_off', 'IRfilter']
        train_img_dir = [f'../../../COSSY/{s}/' for s in videos]
        train_json = [f'../../../COSSY/annotations/{s}.json' for s in videos]
        val_img_dir = '../../../COSSY/Lunch3/'
        val_json = '../../../COSSY/annotations/Lunch3.json'
        lr_SGD = 0.0001 / batch_size / subdivision
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
    elif args.dataset == 'THH1MW':
        videos = ['Meeting1', 'Meeting2', 'Lab2', 'MW']
        train_img_dir = [f'../../../COSSY/{s}/' for s in videos] + ['../../../THEODORE/images']
        train_json = [f'../../../COSSY/annotations/{s}.json' for s in videos] + ['../../../THEODORE/annotations/person_bbox.json']
        val_img_dir = '../../../COSSY/Lab1/'
        val_json = '../../../COSSY/annotations/Lab1.json'
        lr_SGD = 0.0001 / batch_size / subdivision
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
    elif args.dataset == 'THH1H2':
        videos = ['Meeting1', 'Meeting2', 'Lab2',
                  'Lunch1', 'Lunch2', 'Lunch3', 'Edge_cases', 'IRill']
        train_img_dir = [f'../../../COSSY/{s}/' for s in videos] + ['../../../THEODORE/images']
        train_json = [f'../../../COSSY/annotations/{s}.json' for s in videos] + ['../../../THEODORE/annotations/person_bbox.json']
        val_img_dir = '../../../COSSY/Lab1/'
        val_json = '../../../COSSY/annotations/Lab1.json'
        lr_SGD = 0.0001 / batch_size / subdivision
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
    elif args.dataset == 'THH2MW':
        videos = ['Lunch1', 'Lunch2', 'Edge_cases', 'IRill', 'MW']
        train_img_dir = [f'../../../COSSY/{s}/' for s in videos] + ['../../../THEODORE/images']
        train_json = [f'../../../COSSY/annotations/{s}.json' for s in videos] + ['../../../THEODORE/annotations/person_bbox.json']
        val_img_dir = '../../../COSSY/Lunch3/'
        val_json = '../../../COSSY/annotations/Lunch3.json'
        lr_SGD = 0.0001 / batch_size / subdivision
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
    elif args.dataset == 'COCOTHH1MW':
        videos = ['Meeting1', 'Meeting2', 'Lab2', 'MW']
        train_img_dir = ['../../../COCO/train2017', '../../../THEODORE/images']
        train_img_dir += [f'../../../COSSY/{s}/' for s in videos]
        train_json = ['../../../COCO/annotations/instances_train2017.json', '../../../THEODORE/annotations/person_bbox.json']
        train_json += [f'../../../COSSY/annotations/{s}.json' for s in videos]
        val_img_dir = '../../../COSSY/Lab1/'
        val_json = '../../../COSSY/annotations/Lab1.json'
        lr_SGD = 0.0002 / batch_size / subdivision
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
    elif args.dataset == 'Edge_cases':
        # debugging
        train_img_dir = '../../../COSSY/Edge_cases'
        train_json = '../../../COSSY/annotations/Edge_cases.json'
        lr_SGD = 0.0001 / batch_size / subdivision
        # Learning rate setup
        def burnin_schedule(i):
            return 1
    dataset = Dataset4YoloAngle(train_img_dir, train_json, initial_size, enable_aug,
                                only_person=only_person, debug_mode=args.debug)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                            num_workers=num_cpu, pin_memory=True, drop_last=False)
    dataiterator = iter(dataloader)
    
    if args.model == 'ex':
        from models.exact import YOLOv3
        model = YOLOv3(backbone=args.backbone, img_norm=False)
    if args.model == 'ex_LL1':
        from models.exact import YOLOv3
        model = YOLOv3(backbone=args.backbone, loss_angle='LL1', img_norm=False)
    if args.model == 'ex_LL2':
        from models.exact import YOLOv3
        model = YOLOv3(backbone=args.backbone, loss_angle='LL2', img_norm=False)
    if args.model == 'ex_90L1':
        from models.exact import YOLOv3
        model = YOLOv3(backbone=args.backbone, loss_angle='L1', angran=180)
    if args.model == 'ex_90L2':
        from models.exact import YOLOv3
        model = YOLOv3(backbone=args.backbone, loss_angle='L2', angran=180)
    if args.model == 'ex_90pL1':
        from models.exact import YOLOv3
        model = YOLOv3(backbone=args.backbone, loss_angle='period_L1', angran=180)
    if args.model == 'ex_90pL2':
        from models.exact import YOLOv3
        model = YOLOv3(backbone=args.backbone, loss_angle='period_L2', angran=180)
    elif args.model == 'exact_L1':
        from models.exact import YOLOv3
        model = YOLOv3(backbone=args.backbone, img_norm=False, loss_angle='L1')
    elif args.model == 'exact_L2':
        from models.exact import YOLOv3
        model = YOLOv3(backbone=args.backbone, img_norm=False, loss_angle='L2')
    elif args.model == 'ex_pL1':
        from models.exact import YOLOv3
        model = YOLOv3(backbone=args.backbone, img_norm=False,
                       loss_angle='period_L1')
    elif args.model == 'exact_45':
        raise NotImplementedError()
        from models.exact_45 import YOLOv3
        model = YOLOv3()
    elif args.model == 'Hitachi80':
        from models.Hitachi import YOLOv3
        model = YOLOv3(anchor_size='COCO', only_person=False)
    elif args.model == 'Hitachi1':
        from models.Hitachi import YOLOv3
        model = YOLOv3(anchor_size='COCO', only_person=True)
    # elif args.model == 'alpha_fc':
    #     raise Exception('Deprecated')
    #     from models.alphaFC import YOLOv3
    #     model = YOLOv3(anchors=anchors, indices=indices)
    
    model = model.cuda() if args.cuda else model

    start_iter = -1
    # model.apply(weights_init_normal)
    if args.checkpoint:
        print("loading ckpt...", args.checkpoint)
        weights_path = os.path.join('./weights/', args.checkpoint)
        state = torch.load(weights_path)
        model.load_state_dict(state['model_state_dict'])
        start_iter = state['iter']

    val_set = MWtools.MWeval(val_json, iou_method='rle')
    eval_img_names = os.listdir('./images/')
    eval_img_paths = [os.path.join('./images/',s) for s in eval_img_names]
    logger = SummaryWriter(f'./logs/{job_name}')

    # optimizer setup
    params = []
    # set weight decay only on conv.weight
    for key, value in model.named_parameters():
        if 'conv.weight' in key:
            params += [{'params':value, 'weight_decay':decay_SGD}]
        else:
            params += [{'params':value, 'weight_decay':0.0}]

    if args.dataset in {'COCO', 'COCOTHH1MW'}:
        optimizer = torch.optim.SGD(params, lr=lr_SGD, momentum=0.9, dampening=0,
                                    weight_decay=decay_SGD)
    elif args.dataset in {'MW', 'H1H2', 'H1MW', 'H2MW', 'THEODORE', 'THH1MW',
                          'THH1H2', 'THH2MW'}:
        assert args.checkpoint is not None
        optimizer = torch.optim.SGD(params, lr=lr_SGD)
    else:
        raise NotImplementedError()

    if args.dataset not in args.checkpoint:
        start_iter = -1
    else:
        optimizer.load_state_dict(state['optimizer_state_dict'])
        print(f'begin from iteration: {start_iter}')
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, burnin_schedule, last_epoch=start_iter)

    # start training loop
    today = timer.today()
    start_time = timer.tic()
    for iter_i in range(start_iter, 500000):
        # evaluation
        if args.skip_first_eval and iter_i == 0:
            continue
        if iter_i % args.eval_interval == 0 and (args.dataset != 'COCO' or iter_i > 0):
            with timer.contexttimer() as t0:
                model.eval()
                model_eval = api.yoloangle(conf_thres=0.005, model=model)
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
        # loop_times = subdivision if not args.adversarial else subdivision//2
        optimizer.zero_grad()
        for inner_iter_i in range(subdivision):
            try:
                imgs, targets, cats, _, _ = next(dataiterator)  # load a batch
            except StopIteration:
                dataiterator = iter(dataloader)
                imgs, targets, cats, _, _ = next(dataiterator)  # load a batch
            # visualization.imshow_tensor(imgs)
            imgs = imgs.cuda() if args.cuda else imgs
            torch.cuda.reset_max_memory_allocated()
            loss = model(imgs, targets, labels_cats=cats)
            loss.backward()
            # if args.adversarial:
            #     imgs = imgs + imgs.grad*0.05
            #     imgs = imgs.detach()
            #     # visualization.imshow_tensor(imgs)
            #     loss = model(imgs, targets)
            #     loss.backward()
        optimizer.step()
        scheduler.step()

        # logging
        if iter_i % args.print_interval == 0:
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
            torch.cuda.reset_max_memory_allocated(0)

        # random resizing
        if multiscale and iter_i > 0 and (iter_i % multiscale_interval == 0):
            if args.high_resolution:
                imgsize = random.randint(16, 34) * 32
            else:
                low = 10 if args.dataset == 'COCO' else 16
                imgsize = random.randint(low, 21) * 32
            dataset.img_size = imgsize
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                num_workers=num_cpu, pin_memory=True, drop_last=False)
            dataiterator = iter(dataloader)

        # save checkpoint
        if iter_i > 0 and (iter_i % args.checkpoint_interval == 0):
            state_dict = {
                'iter': iter_i,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            save_path = os.path.join('./weights', f'{job_name}_{today}_{iter_i}.ckpt')
            torch.save(state_dict, save_path)

        # save detection
        if iter_i > 0 and iter_i % args.img_interval == 0:
            for img_path in eval_img_paths:
                eval_img = Image.open(img_path)
                dts = api.detect_once(model, eval_img, conf_thres=0.1, input_size=target_size)
                np_img = np.array(eval_img)
                visualization.draw_dt_on_np(np_img, dts)
                np_img = cv2.resize(np_img, (416,416))
                # cv2.imwrite(f'./results/eval_imgs/{job_name}_{today}_{iter_i}.jpg', np_img)
                logger.add_image(img_path, np_img, iter_i, dataformats='HWC')

            model.train()