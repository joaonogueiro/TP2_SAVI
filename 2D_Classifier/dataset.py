#!/usr/bin/env python3

###############################################################
# NOTE Code given by professor Miguel Riem Oliveira in SAVI classes.
# Adapted
###############################################################

import os
import json
import torch
from torchvision import transforms
from PIL import Image

class Dataset(torch.utils.data.Dataset):

    # def __init__(self, filenames, labels_list):
    def __init__(self, filenames):
        self.filenames = filenames
        self.number_of_images = len(self.filenames)
        # self.labels_list = labels_list

        # Load labels_names from json file
        with open('../2D_Classifier/categories.json', 'r') as f:
            dataset_filenames = json.load(f)
        self.labels_dict = dataset_filenames

        # Compute the corresponding labels
        self.labels = [] 
        for filename in self.filenames:
            basename = os.path.basename(filename)
            blocks = basename.split('_')
            # print(blocks)
            if blocks[1].isnumeric():# basename is "instant_noodles_3_4_37_crop.png"
                label = blocks[0]  
            else:
                label = blocks[0] + '_' + blocks[1]
           
            # Pytorch does not accept strings as labels, numeric indexes will be given to each category
            if label in self.labels_dict:
                idx_label = self.labels_dict[label]
                self.labels.append(idx_label)
            else:
                raise ValueError('Unknown label ' + label)
               
        # Create a set of transformations
        # Transformations, can be advantageous when there are few training images
        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)), 
            transforms.ToTensor() 
        ]) # The images must all be the same size on the network
       
    def __len__(self):
        # Must return the size of the data
        return self.number_of_images

    def __getitem__(self, index):
        # Must return the data of the corresponding index
        # Load the image in pil format
        filename = self.filenames[index]
        pil_image = Image.open(filename)

        # Convert to tensor
        tensor_image = self.transforms(pil_image)

        # Get corresponding label
        label = self.labels[index]

        return tensor_image, label
