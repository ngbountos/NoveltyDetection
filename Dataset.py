import os
import random
import cv2 as cv
import numpy as np
import torch
# import torchvision
from torch import nn
from PIL import Image


class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset='../Data/Dataset', mode = 'positive', set='train'):
        self.path = dataset
        self.classes = os.listdir(self.path)
        self.interferograms_normal = []
        self.interferograms_deformation = []
        self.mode = mode
        self.set = set
        for data_class in self.classes:
            images = os.listdir(self.path + '/' + data_class)
            for image in images:
                image_dict = {'path': self.path + '/' + data_class + '/' + image, 'label': data_class}
                if self.mode != 'mixed':
                    if int(data_class) == 0:
                        self.interferograms_normal.append(image_dict)
                    else:
                        self.interferograms_deformation.append(image_dict)
                else:
                    self.interferograms_normal.append(image_dict)
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
        if int(image_label) == 1:
            tmp = np.zeros_like(image)
            image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            tmp[:, :, 0] = image
            tmp[:, :, 1] = image
            tmp[:, :, 2] = image
            image = tmp
            print(image.shape)
            image = image[:225, :225, :]
            cv.imshow('im',image)
            print(image.shape)
            cv.waitKey(0)
        if self.set == 'train':
            angle = random.randint(0,360)

            M = cv.getRotationMatrix2D((113, 113), angle, 1)
            image = cv.warpAffine(image, M, (image.shape[1], image.shape[0]))

        image = np.reshape(image, (image.shape[2], image.shape[0], image.shape[1]))

        image = image/255
        return torch.from_numpy(image).float(), int(image_label)


