import torch
import torchvision
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from utils import randomaug

def encode_cmap(label):
    cmap = colormap()
    return cmap[label.astype(np.int16),:]

def denormalize_img(imgs=None, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]):
    _imgs = torch.zeros_like(imgs)
    _imgs[:,0,:,:] = imgs[:,0,:,:] * std[0] + mean[0]
    _imgs[:,1,:,:] = imgs[:,1,:,:] * std[1] + mean[1]
    _imgs[:,2,:,:] = imgs[:,2,:,:] * std[2] + mean[2]
    _imgs = _imgs.type(torch.uint8)

    return _imgs

def denormalize_img2(imgs=None):
    #_imgs = torch.zeros_like(imgs)
    imgs = denormalize_img(imgs)

    return imgs / 255.0

def minmax_norm(x):
    for i in range(x.shape[0]):
        x[i,...] = x[i,...] - x[i,...].min()
        x[i,...] = x[i,...] / x[i,...].max()
    return x

def colormap(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap


def augment_data_strong(images, n=4, m=20):
    trans_pil = transforms.ToPILImage()
    trans_tensor = transforms.ToTensor()
    norm = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    for i in range(images.shape[0]):
        image_pil_i = trans_pil(images[i])
        aug = randomaug.RandAugment(n, m)
        image_pil_i_aug = trans_tensor(aug(image_pil_i))
        images[i] = norm(image_pil_i_aug)
        images[i] = torch.flip(images[i], dims=[2])

    return images