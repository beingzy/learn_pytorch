"""
"""
import matplotlib.pyplot as plt 
import numpy as np 

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# transforms
def build_transform():
    return transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))])

transforms = build_transform()

train_set = torchvision.datasets.FashionMNIST('./data',
    download=True,
    train=True,
    transform=transforms)
test_set = torchvision.datasets.FashionMNIST('./data',
    download=True,
    train=False,
    transform=transforms)

train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=4,shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(
    test_set, batch_size=4, shuffle=False, num_workers=2)

# constant for classes
classes = (
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')


def display_img(img, one_channel=False):
    """helper function
    """
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5 # unnormalize
    np_img = img.numpy()
    if one_channel:
        plt.imshow(np_img, cmap="Greys")
    else:
        plt.imshow(np.transpose(np_img, (1, 2, 0)))

