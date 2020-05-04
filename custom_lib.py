import numpy as np
import torch
import cv2
import torchvision.transforms.functional as TF
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

import matplotlib.pyplot as plt

class ElasticTransform(object):
    """
    Args:
    """

    def __init__(self, alpha,sigma,seed=None):
        self.alpha = alpha
        self.sigma = sigma
        self.rng = np.random.RandomState(seed)
    def __call__(self, sample):
        image, segmentation = sample['image'], sample['segmentation']

        """`image` is a tensor of shape (M, N) or (1,M,N), same for segmentation"""
        if image.ndim > 2:
            image = image[0]
        if segmentation.ndim > 2:
            segmentation = segmentation[0]

        # Take measurements
        image_shape = image.shape
        ratio = 8
        # Make random fields
        dx = self.rng.uniform(-1, 1,
                (image_shape[0]//ratio,image_shape[1]//ratio))*self.alpha/ratio
        dy = self.rng.uniform(-1, 1,
                (image_shape[0]//ratio,image_shape[1]//ratio))*self.alpha/ratio
        # Smooth dx and dy
        sdx = cv2.GaussianBlur(dx,ksize=(0,0),sigmaX=self.sigma/ratio)
        sdy = cv2.GaussianBlur(dy,ksize=(0,0),sigmaX=self.sigma/ratio)

        #Scale up
        sdx = cv2.resize(sdx, image_shape[::-1])
        sdy = cv2.resize(sdy, image_shape[::-1])
        
        # Make meshgrid
        x, y = np.meshgrid(np.arange(image_shape[1]),np.arange(image_shape[0]))
        # Distort meshgrid indices
        grid_y = (y + sdy).astype(np.float32)
        grid_x = (x + sdx).astype(np.float32)
            
        # Map cooordinates from image to distorted index set
        transformed_image = cv2.remap(image, grid_x, grid_y,
                borderMode=cv2.BORDER_REFLECT_101, interpolation=cv2.INTER_LINEAR)
        transformed_segmentation = cv2.remap(segmentation.astype(np.float32), grid_x, grid_y,
                borderMode=cv2.BORDER_REFLECT_101,interpolation=cv2.INTER_NEAREST)
        return  {"image":transformed_image,
                "segmentation":transformed_segmentation.astype(np.int)}
    
class cToPILImage(object):
    """
    Args:
    """

    def __init__(self, seed=None):
        self.rng = np.random.RandomState(seed)
    def __call__(self, sample):
        image, segmentation = sample['image'], sample['segmentation']
        return {"image":TF.to_pil_image(image),
                "segmentation":segmentation}    

class cColorJitter(object):
    """
    Args:
    """

    def __init__(self,brightness=(1,1),contrast=(1,1), seed=None):
        self.bmin = brightness[0]
        self.bmax = brightness[1]
        self.cmin = contrast[0]
        self.cmax = contrast[1]
        self.rng = np.random.RandomState(seed)
    
    def __call__(self, sample):
        image, segmentation = sample['image'], sample['segmentation']
        brightness = self.rng.uniform(self.bmin,self.bmax) 
        contrast = self.rng.uniform(self.bmin,self.bmax) 
        
        image = TF.adjust_brightness(image,brightness)
        image = TF.adjust_contrast(image,contrast)
        return {"image":image, "segmentation":segmentation}    

class cRandomVerticalFlip(object):
    """
    Args:
    """

    def __init__(self, p=0.5, seed=None):
        self.rng = np.random.RandomState(seed)
    def __call__(self, sample):
        image, segmentation = sample['image'], sample['segmentation']
        if self.rng.uniform(0,1) > 0.5:
            image = TF.vflip(image)
            # Copy is important to avoid negative stride error
            segmentation= np.flipud(segmentation).copy() 
        return {"image":image, "segmentation":segmentation}    

class cRandomHorizontalFlip(object):
    """
    Args:
    """

    def __init__(self,p=0.5, seed=None):
        self.rng = np.random.RandomState(seed)
    def __call__(self, sample):
        image, segmentation = sample['image'], sample['segmentation']
        if self.rng.uniform(0,1) > 0.5:
            image = TF.hflip(image)
            # Copy is important to avoid negative stride error
            segmentation= np.fliplr(segmentation).copy() 
        return {"image":image, "segmentation":segmentation}    

class cToTensor(object):
    """
    Args:
    """

    def __init__(self, seed=None):
        self.rng = np.random.RandomState(seed)
    def __call__(self, sample):
        image, segmentation = sample['image'], sample['segmentation']
        return {"image":TF.to_tensor(image),
                "segmentation":segmentation}    



