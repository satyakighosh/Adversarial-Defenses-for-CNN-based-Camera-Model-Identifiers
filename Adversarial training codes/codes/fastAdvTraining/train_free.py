import argparse
import logging
import os
import time
from tqdm import tqdm
import random

import apex.amp as amp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import densenet121 as _densenet

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms, utils


import random
# from sklearn.feature_extraction import image
from PIL import Image

from utils import (IMG_SIZE, attack_pgd, patch_and_image_level_accuracy, upper_limit, lower_limit, std, clamp, get_loaders,
    evaluate_pgd, evaluate_standard, set_parameter_requires_grad, initialize_model)

logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--data-dir', default='./training_data.npy', type=str)
    parser.add_argument('--epochs', default=80, type=int, help='Total number of epochs will be this argument * number of minibatch replays.')
    parser.add_argument('--lr-schedule', default='cyclic', type=str, choices=['cyclic', 'multistep'])
    parser.add_argument('--lr-min', default=0., type=float)
    parser.add_argument('--lr-max', default=0.004, type=float)
    parser.add_argument('--weight-decay', default=5e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--minibatch-replays', default=8, type=int)
    parser.add_argument('--out-dir', default='train_free_output', type=str, help='Output directory')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--opt-level', default='O2', type=str, choices=['O0', 'O1', 'O2'],
        help='O0 is FP32 training, O1 is Mixed Precision, and O2 is "Almost FP16" Mixed Precision')
    parser.add_argument('--loss-scale', default='1.0', type=str, choices=['1.0', 'dynamic'],
        help='If loss_scale is "dynamic", adaptively adjust the loss scale over time')
    parser.add_argument('--master-weights', action='store_true',
        help='Maintain FP32 master weights to accompany any FP16 model weights, not applicable for O1 opt level')
    return parser.parse_args()


def main():
    args = get_args()

    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    logfile = os.path.join(args.out_dir, 'output.log')
    if os.path.exists(logfile):
        os.remove(logfile)

    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        filename=logfile)
    logger.info(args)

    epsilon = (args.epsilon / 255.) / std
    pgd_alpha = (2 / 255.) / std
    IMG_SIZE = 224
    num_classes = 5
    model_name = "densenet"

    # Flag for feature extracting. When False, we finetune the whole model,
    #   when True we only update the reshaped layer params
    feature_extract = False 

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    train_loader, test_loader = get_loaders(args.data_dir, args.batch_size)

    # model = _densenet(pretrained=True).cuda()
    # Initialize the model for this run
    model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
    model = model_ft.cuda()

    # Print the model we just instantiated
    # print(model_ft)

    model.train()

    opt = torch.optim.SGD(model.parameters(), lr=args.lr_max, momentum=args.momentum, weight_decay=args.weight_decay)
    amp_args = dict(opt_level=args.opt_level, loss_scale=args.loss_scale, verbosity=False)
    if args.opt_level == 'O2':
        amp_args['master_weights'] = args.master_weights
    model, opt = amp.initialize(model, opt, **amp_args)
    criterion = nn.CrossEntropyLoss()

    delta = torch.zeros(args.batch_size, 3, IMG_SIZE, IMG_SIZE).cuda()
    delta.requires_grad = True

    lr_steps = args.epochs * len(train_loader) * args.minibatch_replays
    if args.lr_schedule == 'cyclic':
        scheduler = torch.optim.lr_scheduler.CyclicLR(opt, base_lr=args.lr_min, max_lr=args.lr_max,
            step_size_up=lr_steps / 2, step_size_down=lr_steps / 2)
    elif args.lr_schedule == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[lr_steps / 2, lr_steps * 3 / 4], gamma=0.1)

    # Training
    print("Starting training...")
    start_train_time = time.time()
    logger.info('Epoch \t Seconds \t LR \t \t Train Loss \t Train Acc \t Robust Acc')
    for epoch in range(args.epochs):
        start_epoch_time = time.time()
        train_loss = 0
        train_acc = 0
        train_n = 0
        loop = tqdm(enumerate(train_loader), total = len(train_loader))
        for i, (X, y) in loop:
            X, y = X.cuda(), y.cuda()
            if i == 0:
                first_batch = (X, y)

            # Adversarial_training:    
            for _ in range(args.minibatch_replays):
                output = model(X + delta[:X.size(0)])
                loss = criterion(output, y)
                opt.zero_grad()
                with amp.scale_loss(loss, opt) as scaled_loss:
                    scaled_loss.backward()
                grad = delta.grad.detach()
                delta.data = clamp(delta + epsilon * torch.sign(grad), -epsilon, epsilon)
                delta.data[:X.size(0)] = clamp(delta[:X.size(0)], lower_limit - X, upper_limit - X)
                opt.step()
                delta.grad.zero_()
                scheduler.step()


            train_loss += loss.item() * y.size(0)
            train_acc += (output.max(1)[1] == y).sum().item()
            train_n += y.size(0)
            
            # update progress bar
            loop.set_description(f"Epoch [{epoch} / {args.epochs}]")
            loop.set_postfix(loss = train_loss/train_n, acc = train_acc/train_n)

        # Check current PGD robustness of model using random minibatch
        X, y = first_batch
        pgd_delta = attack_pgd(model, X, y, epsilon, pgd_alpha, 5, 1, opt)
        with torch.no_grad():
            output = model(clamp(X + pgd_delta[:X.size(0)], lower_limit, upper_limit))
        robust_acc = (output.max(1)[1] == y).sum().item() / y.size(0)
        print(f"Robust accuracy: {robust_acc}")

        epoch_time = time.time()
        lr = scheduler.get_lr()[0]
        logger.info('%d \t %.1f \t \t %.4f \t %.4f \t %.4f \t %.4f',
            epoch, epoch_time - start_epoch_time, lr, train_loss/train_n, train_acc/train_n, robust_acc)
        
        if epoch%5 == 0: 
            patch_and_image_level_accuracy(model=model)

        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'loss': train_loss/train_n,
                }, os.path.join(args.out_dir, f'model_{epoch}.pth'))
    train_time = time.time()
    # torch.save(model.state_dict(), os.path.join(args.out_dir, 'model.pth'))
    # torch.save({
    #         'epoch': epoch,
    #         'model_state_dict': model.state_dict(),
    #         'optimizer_state_dict': opt.state_dict(),
    #         'loss': train_loss/train_n,
    #         }, os.path.join(args.out_dir, f'model_{epoch}.pth'))
    logger.info('Total train time: %.4f minutes', (train_time - start_train_time)/60)

    # Evaluation
    print('Starting evaluation...')
    # model_test = _densenet(pretrained=False).cuda()
    model_test = model
    model_test.load_state_dict(model.state_dict())
    model_test.float()
    model_test.eval()

    pgd_loss, pgd_acc = evaluate_pgd(test_loader, model_test, 50, 10)
    test_loss, test_acc = evaluate_standard(test_loader, model_test)

    logger.info('Test Loss \t Test Acc \t PGD Loss \t PGD Acc')
    logger.info('%.4f \t \t %.4f \t %.4f \t %.4f', test_loss, test_acc, pgd_loss, pgd_acc)


if __name__ == "__main__":
    # try getting GPU until it is done
    ran = False
    while ran == False:
        try:
            main()
            ran = True
        except RuntimeError:
            None


# try smaller lr to prevent the sudden drop in accuracy
# the sudden drop is absent when no adversarial training is done
# try densenet201
# try batch of 32 OR 128
# vary epochs
# to normalize or not? yes
# FIXED: why perturbations in the order of e-25 and still getting misclassified? Densenet was predicting 1000 classes