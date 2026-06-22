import torch
import numpy as np
from tqdm import tqdm

#Added
import sys
from torchvision import transforms
from torch import tensor
from sklearn import random_projection
from typing import Tuple
#import timm
from PIL import ImageFilter

class GaussianBlur:
    """Class for Gaussian Blurring of the patch scores tensor"""

    def __init__(self, radius : int = 4):
        self.radius = radius
        self.unload = transforms.ToPILImage()
        self.load = transforms.ToTensor()
        self.blur_kernel = ImageFilter.GaussianBlur(radius=4)

    def __call__(self, img):
        map_max = img.max()
        final_map = self.load(
            self.unload(img[0]/map_max).filter(self.blur_kernel)
        )*map_max
        return final_map