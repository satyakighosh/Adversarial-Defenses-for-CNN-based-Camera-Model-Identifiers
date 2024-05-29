import os
import random
import warnings

import apex.amp as amp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms, utils, models
from tqdm import tqdm

import cv2
import random
from sklearn.feature_extraction import image
from PIL import Image

# Ignore warnings
# import warnings
warnings.filterwarnings("ignore")

vision_mean = (0.4333, 0.4635, 0.4767)
vision_std = (0.2687, 0.2536, 0.2593)

mu = torch.tensor(vision_mean).view(3,1,1).cuda()
std = torch.tensor(vision_std).view(3,1,1).cuda()

upper_limit = ((1 - mu)/ std)
lower_limit = ((0 - mu)/ std)

root_dir = '/home/mtech/2020/satyaki_ghosh/codes/fastAdvTraining/fast_adversarial/VISION'
image_dir = '/home/mtech/2020/satyaki_ghosh/datasets/VISION'
IMG_SIZE = 224
channels = 3

# LOADING THE DATSET
with open(os.path.join(root_dir, 'training_data.npy'),'rb') as f:
    dummy = np.load(f, allow_pickle=True)
dataset = list(dummy)
random.shuffle(dataset)
split_ratio = 0.8
split_num = int(len(dataset) * split_ratio)
trainset = dataset[0 : split_num]
testset = dataset[split_num : ]


class VISIONDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, train=True, root_dir=root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.trainset = trainset
        self.testset = testset
        self.root_dir = root_dir
        self.transform = transform
        self.train = train

    def __len__(self):
        if self.train:
            return len(self.trainset)
        else: 
            return len(self.testset)

    def __getitem__(self, idx):

        # print(f'Index1: {idx}')
        if torch.is_tensor(idx):            
            idx = idx.tolist()

        # CREATING INPUT AND LABELS
        X = []
        y = []

        # print(f'Index2: {idx}')

        if self.train==True:
            X = self.trainset[idx][0]
            y = self.trainset[idx][1]

        else:
            X = self.testset[idx][0]
            y = self.testset[idx][1]         


        X = np.array(X)
        y = np.array(y)

        if self.transform:
            X = self.transform(X)

        return (X, y)


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


def get_loaders(dir_, batch_size):
    train_transform = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(vision_mean, vision_std),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(vision_mean, vision_std),
    ])
    num_workers = 2
    train_dataset = VISIONDataset(
        train=True, transform=train_transform)
    test_dataset = VISIONDataset(
        train=False, transform=test_transform)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers
    )
    return train_loader, test_loader


def attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts, opt=None):
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for zz in range(restarts):
        delta = torch.zeros_like(X).cuda()
        for i in range(len(epsilon)):
            delta[:, i, :, :].uniform_(-epsilon[i][0][0].item(), epsilon[i][0][0].item())
        delta.data = clamp(delta, lower_limit - X, upper_limit - X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(X + delta)
            index = torch.where(output.max(1)[1] == y)
            if len(index[0]) == 0:
                break            
            loss = F.cross_entropy(output, y)
            if opt is not None:
                with amp.scale_loss(loss, opt) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            grad = delta.grad.detach()
            d = delta[index[0], :, :, :]
            g = grad[index[0], :, :, :]
            d = clamp(d + alpha * torch.sign(g), -epsilon, epsilon)
            d = clamp(d, lower_limit - X[index[0], :, :, :], upper_limit - X[index[0], :, :, :])
            delta.data[index[0], :, :, :] = d
            delta.grad.zero_()
        all_loss = F.cross_entropy(model(X+delta), y, reduction='none').detach()
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta


def evaluate_pgd(test_loader, model, attack_iters, restarts):
    epsilon = (8 / 255.) / std
    alpha = (2 / 255.) / std
    pgd_loss = 0
    pgd_acc = 0
    n = 0
    model.eval()
    loop = tqdm(enumerate(test_loader), total = len(test_loader))
    for i, (X, y) in loop:
        X, y = X.cuda(), y.cuda()
        pgd_delta = attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts)
        with torch.no_grad():
            output = model(X + pgd_delta)
            loss = F.cross_entropy(output, y)
            pgd_loss += loss.item() * y.size(0)
            pgd_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
        loop.set_description("PGD evaluation:")
    return pgd_loss/n, pgd_acc/n


def evaluate_standard(test_loader, model):
    test_loss = 0
    test_acc = 0
    n = 0
    model.eval()
    with torch.no_grad():
        loop = tqdm(enumerate(test_loader), total = len(test_loader))
        for i, (X, y) in loop:
            X, y = X.cuda(), y.cuda()
            output = model(X)
            loss = F.cross_entropy(output, y)
            test_loss += loss.item() * y.size(0)
            test_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
            loop.set_description("Standard evaluation:")
    return test_loss/n, test_acc/n


def individual_evaluate_pgd(X, y, model, attack_iters, restarts):
    epsilon = (8 / 255.) / std   
    alpha = (2 / 255.) / std
    pgd_loss = 0
    pgd_acc = 0
    n = 0
    count = 0
    model.eval()
    X, y = X.cuda(), y.cuda()
    pgd_delta = attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts)
    with torch.no_grad():
        output = model(X + pgd_delta)
        loss = F.cross_entropy(output, y)
        pgd_loss += loss.item() * y.size(0)
        pgd_acc += (output.max(1)[1] == y).sum().item()
        n += y.size(0)
    return pgd_loss/n, pgd_acc/n, output

def individual_evaluate_standard(X, y, model):
    test_loss = 0
    test_acc = 0
    n = 0
    model.eval()
    with torch.no_grad():
        X, y = X.cuda(), y.cuda()
        output = model(X)
        loss = F.cross_entropy(output, y)
        test_loss += loss.item() * y.size(0)
        test_acc += (output.max(1)[1] == y).sum().item()
        n += y.size(0)
    return test_loss/n, test_acc/n, output

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False



def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    """ Densenet
    """
    model_ft = models.densenet121(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.classifier.in_features
    model_ft.classifier = nn.Linear(num_ftrs, num_classes)
    input_size = 224

    return model_ft, input_size

def initialize_model_201(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    """ Densenet
    """
    model_ft = models.densenet201(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.classifier.in_features
    model_ft.classifier = nn.Linear(num_ftrs, num_classes)
    input_size = 224

    return model_ft, input_size


def patch_and_image_level_accuracy(model):
    patches_per_photo = 50
    images_per_model = 5

    CATEGORIES = [
        "D01_Samsung_GalaxyS3Mini",
        "D02_Apple_iPhone4s",
        "D03_Huawei_P9",
        "D04_LG_D290",
        "D05_Apple_iPhone5c"]

    total = 0
    correct_images_normal = 0
    correct_patches_normal = 0
    correct_images_adv = 0
    correct_patches_adv = 0


    for camera_model in CATEGORIES:
        y = CATEGORIES.index(camera_model)
        path = os.path.join(os.path.join(image_dir, camera_model), 'images/nat')

        # shuffling the images to avoid similar/close images
        photos = os.listdir(path)
        random.shuffle(photos)
        test_images = photos[0:images_per_model]

        loop = tqdm(test_images, total = len(test_images))
        for image_name in loop:
            total += 1
            image_path = os.path.join(path, image_name)
            img = cv2.imread(image_path)

            pgd_preds = []
            std_preds = []

            patches = image.extract_patches_2d(img, (IMG_SIZE, IMG_SIZE), max_patches = patches_per_photo)
            y = torch.tensor(y, dtype=torch.long).reshape(1).cuda()

            for i in range(patches_per_photo) :                    
                X = patches[i]            
                X = X.astype('float')
                X = torch.from_numpy(X)
                X = X.permute(2, 0, 1)
                X = X.reshape(-1,3,224,224).float()
                test_loss, test_acc, std_output = individual_evaluate_standard(X, y, model)
                std_preds.append(std_output.max(1)[1])
                pgd_loss, pgd_acc, pgd_output = individual_evaluate_pgd(X, y, model, 50, 10)
                pgd_preds.append(pgd_output.max(1)[1])
                if y == std_output.max(1)[1]:
                    correct_patches_normal = correct_patches_normal + 1
                if y[0] == pgd_output.max(1)[1][0]:
                    correct_patches_adv = correct_patches_adv + 1
                    
            normal_output = max(set(std_preds), key=std_preds.count)[0]
            adv_output = max(set(pgd_preds), key=pgd_preds.count)[0]

            if normal_output == y[0]:
                correct_images_normal = correct_images_normal + 1
            if adv_output == y[0]:
                correct_images_adv = correct_images_adv + 1            

            loop.set_postfix(img_acc = correct_images_normal / total,
                             patch_acc = correct_patches_normal / (total * patches_per_photo),
                             adv_img_acc = correct_images_adv / total,
                             adv_patch_acc = correct_patches_adv / (total * patches_per_photo)
                            )
            
    print(f'Image-level normal accuracy: {correct_images_normal / total}')
    print(f'Patch-level normal accuracy: {correct_patches_normal / (total * patches_per_photo)}')
    print()
    print(f'Image-level robust accuracy: {correct_images_adv / total}')
    print(f'Patch-level robust accuracy: {correct_patches_adv / (total * patches_per_photo)}')