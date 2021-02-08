import os
import random

import cv2 as cv
import numpy as np
import torch
# import torchvision
from torch import nn
from PIL import Image


class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset='../Data/Dataset', mode = 'positive'):
        self.path = dataset
        self.classes = os.listdir(self.path)
        self.interferograms_normal = []
        self.interferograms_deformation = []
        self.mode = mode
        for data_class in self.classes:
            images = os.listdir(self.path + '/' + data_class)
            for image in images:
                image_dict = {'path': self.path + '/' + data_class + '/' + image, 'label': data_class}
                if int(data_class) == 0:
                    self.interferograms_normal.append(image_dict)
                else:
                    self.interferograms_deformation.append(image_dict)
        if self.mode == 'positive':
            self.num_examples = len(self.interferograms_deformation)
        else:
            self.num_examples = len(self.interferograms_normal)

    def __len__(self):
        return self.num_examples

    def __getitem__(self, index):
        if self.mode == 'positive':
            image_data = self.interferograms_deformation[index]
        else:
            image_data = self.interferograms_normal[index]
        image_file = image_data['path']
        image_label = image_data['label']
        image = cv.imread(image_file)
        image = image[:226, :226, :]

        image = np.reshape(image, (image.shape[2], image.shape[0], image.shape[1]))

        return torch.from_numpy(image).float(), int(image_label)


