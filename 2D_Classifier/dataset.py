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

        # Reading labels_names from json file
        with open('../Train/dataset_filenames.json', 'r') as f:
            dataset_filenames = json.load(f)
        self.labels_list = dataset_filenames['labels_names']

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
           
            # labels_list -> ['instant_noodles', 'camera', 'sponge', 'lemon', 'stapler', 'cereal_box', 'cap', 'greens', 'orange', ..
            # 'potato', 'glue_stick', 'bowl', 'water_bottle', 'garlic', 'binder', 'pitcher', 'bell_pepper', 'tomato', 'onion', 'food_cup', 
            # 'keyboard', 'calculator', 'soda_can', 'comb', 'food_bag', 'coffee_mug', 'pliers', 'kleenex', 'lightbulb', 'toothbrush', 'plate', 
            # 'banana', 'toothpaste', 'scissors', 'notebook', 'peach', 'dry_battery', 'rubber_eraser', 'marker', 'pear', 'shampoo', 'food_jar', 
            # 'lime', 'cell_phone', 'hand_towel', 'mushroom', 'ball', 'food_box', 'apple', 'flashlight', 'food_can']

        # Pytorch does not accept strings as labels, numeric indexes will be given to each category
            if label in self.labels_list:
                idx_label = self.labels_list.index(label)
                self.labels.append(idx_label)
            else:
                raise ValueError('Unknown label ' + label)
               
        # Debug
        # print(self.filenames[0:3])
        # print(self.labels[0:3]) 
        # Example:
        # filenames ['../../rgbd-dataset/water_bottle/water_bottle_10/water_bottle_10_2_27_crop.png', '../../rgbd-dataset/sponge/sponge_6/sponge_6_4_50_crop.png', '../../rgbd-dataset/soda_can/soda_can_1/soda_can_1_1_117_crop.png']
        # labels [12, 2, 22]

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
