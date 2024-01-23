#!/usr/bin/env python3

import os
import glob
import torch
import json
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from dataset import Dataset
from torchvision import transforms
from model import Model
from trainer import Trainer

def main():
    # -----------------------------------------------------------------
    # Hyperparameters Initialization
    # -----------------------------------------------------------------
    batch_size = 200 # "Sample"
    learning_rate = 0.001
    num_epochs = 10 # Number of epoch (passes through the dataset)

    # -----------------------------------------------------------------
    # Create Model
    # -----------------------------------------------------------------
    model = Model()

    # -----------------------------------------------------------------
    # Prepare Dataset
    # -----------------------------------------------------------------
    
    # dataset_path = '../../rgbd-dataset' 
    # image_filenames = glob.glob(dataset_path + '/*/*/*_crop.png')

    # # Only select k images (Random) to do the train (Speed up)
    # image_filenames = random.sample(image_filenames, k = 1000)
    
    # # Save the name of the categories by the name of the folders
    # if os.path.isdir(dataset_path):
    #     labels_list = [name for name in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, name))]
    # else:
    #     print(f'ERROR The path not exist: ({dataset_path}).')

    # # print(image_filenames)
    # # print(f'Object categories: {labels_list}')
    # # print(f'The dataset have {(len(image_filenames))} images and {len(labels_list)} categories')

    # # Separation of images from the database for training (function of library - sklearn.model_selection )
    # train_filenames, validation_filenames = train_test_split(image_filenames, test_size=0.3) #80% for training

    # print('We have a total of ' + str(len(image_filenames)) + ' images. Used '
    #       + str(len(train_filenames)) + ' for training and ' + str(len(validation_filenames)) +
    #       ' for validation.')
   
    with open('../Train/dataset_filenames.json', 'r') as f:
        dataset_filenames = json.load(f)

    train_filenames = dataset_filenames['train_filenames']
    validation_filenames = dataset_filenames['validation_filenames']

    # Speed up the process
    train_filenames = train_filenames[0:1000]
    validation_filenames = validation_filenames[0:200]

    print('Used ' + str(len(train_filenames)) + ' for training and ' + str(len(validation_filenames)) +
          ' for validation.')
    
    train_dataset = Dataset(train_filenames)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    
    validation_dataset = Dataset(validation_filenames)
    validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=True)

    # Just for testing the train_loader
    # tensor_to_pil_image = transforms.ToPILImage()
    # for batch_idx, (inputs, labels) in enumerate(train_loader):
    #     print('batch_idx = ' + str(batch_idx))
    #     print('inputs shape = ' + str(inputs.shape))

    #     model.forward(inputs)

    #     image_tensor_0 = inputs[0, :, :, :]
    #     print(image_tensor_0.shape)

    #     image_pil_0 = tensor_to_pil_image(image_tensor_0)
    #     print(type(image_pil_0))

    #     fig = plt.figure()
    #     plt.imshow(image_pil_0)
    #     plt.show()
    #     exit(0)

    # -----------------------------------------------------------------
    # Train
    # -----------------------------------------------------------------
    # trainer = Trainer(model, train_loader, learning_rate)
    trainer = Trainer(model=model,
                      train_loader=train_loader,
                      validation_loader = validation_loader,
                      learning_rate=learning_rate,
                      num_epoch = num_epochs,
                      model_path='models/checkpoint.pkl',
                      load_model=True)
    trainer.train()
    
    plt.show()

if __name__ == "__main__":
    main()
