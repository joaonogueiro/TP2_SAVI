#!/usr/bin/env python3

import json
import torch
from model import Model
from colorama import Fore, Style
from torchvision import transforms

class Classifier_2D():

    def __init__(self, image_input):
        
        # Device (PyTorch)
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        # Model         
        model_path = '../2D_Classifier/models/checkpoint.pkl'
        self.model = Model()
        load_model = torch.load(model_path)
        self.model.load_state_dict(load_model['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
                
        self.PIL_to_Tensor = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])

        with open('../2D_Classifier/categories.json', 'r') as f:
            dataset_filenames = json.load(f)
        self.labels_dict = dataset_filenames

        self.image_pill = image_input

    def predicted(self):

        image_transform = self.PIL_to_Tensor(self.image_pill)
        image_transform = image_transform.unsqueeze(0)

        image_transform = image_transform.to(self.device)
        label_t_predicted = self.model.forward(image_transform)

        prediction = torch.argmax(label_t_predicted)
        label = prediction.data.item()
        # print(label)
                
        predicted_label = next(key for key, value in self.labels_dict.items() if value == label)
        
        return predicted_label


    
    
