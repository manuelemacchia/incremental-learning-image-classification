import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

import torch
import torchvision

def image_grid(images, one_channel=False):
    """Show an image grid.
    
    Args:
        images (torch.Tensor): batch of images to show
        one_channel (bool): if True, show images in grayscale, otherwise
            show images in RGB (in this case, images tensor should contain
            three channels per image)
    """
    img_grid = torchvision.utils.make_grid(images)

    if one_channel:
        img_grid = img_grid.mean(dim=0)
    img_grid = img_grid / 2 + 0.5 # unnormalize
    npimg = img_grid.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))