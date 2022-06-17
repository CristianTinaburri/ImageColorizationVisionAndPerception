import torch
import os
from torch.utils.data import Dataset
from torchvision.io import read_image
import cv2
from skimage import color
import numpy as np

# DATASET GENERATOR
class ImageDataset(Dataset):
    def __init__(self, img_dir, test=False, transform=None):
        self.test = test
        self.img_dir = img_dir
        self.images = self.get_images()
        self.transform = transform

    def get_images(self):
      images = []
      for image in os.listdir(self.img_dir):
        images.append(image)
      return images
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])

        image = cv2.imread(img_path)

        if self.transform:
          image = self.transform(image)

        lab_image = color.rgb2lab(np.array(image))
        lab_image = np.transpose(lab_image, (2,0,1))

        l_space = lab_image[0:1]
        ab_space = lab_image[1:3]

        return torch.FloatTensor(l_space), torch.FloatTensor(ab_space)

    def get_rgb_image(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        image = cv2.imread(img_path)

        if self.transform:
          image = self.transform(image)
        
        return torch.FloatTensor(image)