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
    batch_size = 100 # "Sample" #100
    learning_rate = 0.001
    max_num_epochs = 60 # Number of epoch (passes through the dataset) #20
    # loss_threshold = 0.001

    # -----------------------------------------------------------------
    # Create Model
    # -----------------------------------------------------------------
    model = Model()

    # -----------------------------------------------------------------
    # Prepare Dataset
    # -----------------------------------------------------------------
    
    with open('../2D_Classifier/dataset_filenames.json', 'r') as f:
        dataset_filenames = json.load(f)

    train_filenames = dataset_filenames['train_filenames']
    # validation_filenames = dataset_filenames['validation_filenames'] # Não vai ser usada, em principio

    print(f'Used {len(train_filenames)} for training')

    # print('Used ' + str(len(train_filenames)) + ' for training and ' + str(len(validation_filenames)) +
    #       ' for validation.')
    
    train_dataset = Dataset(train_filenames)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    
    # validation_dataset = Dataset(validation_filenames)
    # validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=True)

    # -----------------------------------------------------------------
    # Train
    # -----------------------------------------------------------------
    # trainer = Trainer(model, train_loader, learning_rate)
    trainer = Trainer(model=model,
                      train_loader=train_loader,
                      learning_rate=learning_rate,
                      num_epochs = max_num_epochs,
                      model_path='models/checkpoint.pkl',
                      load_model=True)
    
    trainer.train()
    
    plt.show()

if __name__ == "__main__":
    main()
