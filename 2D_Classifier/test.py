#!/usr/bin/env python3

import json
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from dataset import Dataset
from model import Model
from sklearn import metrics

def main():
    # -----------------------------------------------------------------
    # Create model
    # -----------------------------------------------------------------
    model = Model()

    # -----------------------------------------------------------------
    # Prepare Datasets
    # -----------------------------------------------------------------
    with open('dataset_filenames.json', 'r') as f:
        # Reading from json file
        dataset_filenames = json.load(f)

    test_filenames = dataset_filenames['test_filenames']
    print(f'Test {len(test_filenames)} images')
    test_filenames = test_filenames[0:2000]
    # test_filenames = random.sample(test_filenames, 2000)
    print('Used ' + str(len(test_filenames)) + ' for testing ')

    test_dataset = Dataset(test_filenames)

    batch_size = len(test_filenames)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    # -----------------------------------------------------------------
    # Prediction
    # -----------------------------------------------------------------
    # device = 'cuda' if torch.cuda.is_available() else 'cpu' #Doesn't work idk
    device = 'cpu'
    
    # Load the trained model
    checkpoint = torch.load('models/checkpoint.pkl')
    model.load_state_dict(checkpoint['model_state_dict'])

    model.to(device)
    model.eval()  # we are in testing mode

    predicted_is = []
    labels_gt_np = []

    for batch_idx, (inputs, labels_gt) in enumerate(test_loader):
        # move tensors to device
        inputs = inputs.to(device)
        labels_gt = labels_gt.to(device)

        # Get predicted labels
        labels_predicted = model.forward(inputs)

        # Transform predicted labels into probabilities
        predicted_probabilities = F.softmax(labels_predicted, dim=1).tolist()

        # Make a decision using the largest probability
        for i in predicted_probabilities:
            idx = i.index(max(i))
            predicted_is.append(idx)

        labels_gt_np = labels_gt.cpu().detach().numpy()

    # Accuracy = metrics.accuracy_score(labels_gt_np, predicted_is)
    Precision = metrics.precision_score(labels_gt_np, predicted_is, average='weighted')
    Sensitivity_recall = metrics.recall_score(labels_gt_np, predicted_is, average='weighted')
    F1_score = metrics.f1_score(labels_gt_np, predicted_is, average='weighted')
    print(f'Precision = {Precision}') # In the simplest terms, Precision is the ratio between the True Positives and all the points that are classified as Positives.
    print(f'Recall = {Sensitivity_recall}') # To put it simply, Recall is the measure of our model correctly identifying True Positives. It is also called a True positive rate.
    print(f'F1_score = {F1_score}') # F1-score is the Harmonic mean of the Precision and Recall

    confusion_matrix = metrics.confusion_matrix(labels_gt_np, predicted_is)

    # Normalizar a matriz de confusão manualmente
    normalized_confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]

    # Criar ConfusionMatrixDisplay com a matriz normalizada
    categories = ['bowl', 'cap', 'cereal_box', 'coffee_mug', 'soda_can'] 
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=normalized_confusion_matrix, display_labels=categories)

    # Exibir a matriz de confusão
    cm_display.plot(cmap='viridis', values_format='.2f')  # Não é necessário 'normalize' aqui

    plt.show()

if __name__ == "__main__":
    main()
