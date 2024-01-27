#!/usr/bin/env python3

import json
import torch
from torchvision import transforms
import matplotlib.pyplot as plt

from dataset import Dataset
from model import Model
import torch.nn.functional as F


def metrics(ground_truth, predicted, l):
    # Count FP, FN, TP, and TN
    TP, FP, TN, FN = 0, 0, 0, 0
    for gt, pred in zip(ground_truth, predicted):

        if gt == l and pred == l:  # True positive
            TP += l
        elif gt != l and pred == l:  # False positive
            FP += 1
        elif gt == l and pred == l:  # False negative
            FN += 1
        elif gt == l and pred == l:  # True negative
            TN += 1

    # Compute precision and recall
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * (precision*recall)/(precision+recall)

    return TP, FP, FN, TN, precision, recall, f1_score


def main():
    # -----------------------------------------------------------------
    # Hyperparameters initialization
    # -----------------------------------------------------------------
    learning_rate = 0.001
    num_epochs = 50

    # -----------------------------------------------------------------
    # Create model
    # -----------------------------------------------------------------
    model = Model()

    # -----------------------------------------------------------------
    # Prepare Datasets
    # -----------------------------------------------------------------
    with open('dataset_filenames.json', 'r') as f:
        dataset_filenames = json.load(f)

    with open('categories.json', 'r') as f:
        categories_dict = json.load(f)

    test_filenames = dataset_filenames['test_filenames']
    # test_filenames = test_filenames[0:100]

    labels_names = categories_dict

    print('Used ' + str(len(test_filenames)) + ' for testing ')

    test_dataset = Dataset(test_filenames)

    batch_size = len(test_filenames)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    # Just for testing the train_loader
    tensor_to_pil_image = transforms.ToPILImage()

    # -----------------------------------------------------------------
    # Prediction
    # -----------------------------------------------------------------
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # Load the trained model
    checkpoint = torch.load('models/checkpoint.pkl')
    model.load_state_dict(checkpoint['model_state_dict'])

    model.to(device)
    model.eval()  # we are in testing mode
    batch_losses = []
    for batch_idx, (inputs, labels_gt) in enumerate(test_loader):

        # move tensors to device
        inputs = inputs.to(device)
        labels_gt = labels_gt.to(device)

        # Get predicted labels
        labels_predicted = model.forward(inputs)

    # Transform predicted labels into probabilities
    predicted_probabilities = F.softmax(labels_predicted, dim=1).tolist()

    
    predicted_is = []
    variables = ["bowl", "cap", "cereal", "coffee", "soda"]
    # Make a decision using the largest probability
    for i in predicted_probabilities:
        idx = i.index(max(i))
        # predicted_is.append(variables[idx])
        predicted_is.append(idx)


    labels_gt_np = labels_gt.cpu().detach().numpy()

    # Show Images
    labels_names = {value: key for key, value in labels_names.items()}
    fig = plt.figure()
    idx_image = 0
    for row in range(4):
        for cols in range(4):
            image_tensor = inputs[idx_image, :, :, :]
            image_pil = tensor_to_pil_image(image_tensor)

            ax = fig.add_subplot(4,4,idx_image+1)
            plt.imshow(image_pil)

            # Remove axis
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
            ax.xaxis.set_ticks([])
            ax.yaxis.set_ticks([])

            ground_truth = labels_gt_np[idx_image]
            predicted = predicted_is[idx_image]

            if ground_truth == predicted:
                color = 'green'
            else:
                color = 'red'

            text = labels_names.get(predicted)
            # Put labels
            ax.set_xlabel(text, color = color)

            idx_image +=1

    plt.show()

if __name__ == "__main__":
    main()
