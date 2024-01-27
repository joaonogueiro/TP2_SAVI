#!/usr/bin/env python3

import os
import glob
import json
import random
from sklearn.model_selection import train_test_split

# This program ensures that the images that are used for the 
#various processes (training, validation and testing) are not repeated

def categoriesNames(dataset_path):
    # Save the name of the categories by the name of the folders
    if os.path.isdir(dataset_path):
        labels_list = [name for name in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, name))]
        classes = {labels_list: i for i, labels_list in enumerate(labels_list)}
        json_categories = json.dumps(classes, indent=2)
        with open("categories.json", "w") as outfile:
            outfile.write(json_categories)
    else:
        print(f'ERROR The path not exist: ({dataset_path}).')
    return labels_list
       
       
def main():

    dataset_path = '../../rgbd-dataset' 
    image_filenames = glob.glob(dataset_path + '/*/*/*_crop.png')

    # Check if dataset data exists
    if len(image_filenames) < 1:
        raise FileNotFoundError('Dataset files not found')
    
    # Speed up
    image_filenames = random.sample(image_filenames, k=2000)
    
    
    # Splitting the dataset
    # train_filenames, remaining_filenames = train_test_split(image_filenames, test_size=0.3)
    # validation_filenames, test_filenames = train_test_split(remaining_filenames, test_size=0.33)

    # print('We have a total of ' + str(len(image_filenames)) + ' images.')
    # print(f'Used {len(train_filenames)} train images')
    # print(f'Used {len(validation_filenames)} validation images')
    # print(f'Used {len(test_filenames)} test images')

    # labels_list = categoriesNames(dataset_path)

    # d = {'train_filenames': train_filenames,
    #      'validation_filenames': validation_filenames,
    #      'test_filenames': test_filenames,
    #      'labels_names': labels_list}

    # Slipt the Database 80% - Train and 20% - Test 
    train_filenames, test_filenames = train_test_split(image_filenames, test_size=0.2)

    labels_list = categoriesNames(dataset_path)

    d = {'train_filenames': train_filenames,
         'test_filenames': test_filenames}

    json_object = json.dumps(d, indent=2)

    # Writing to sample.json
    with open("dataset_filenames.json", "w") as outfile:
        outfile.write(json_object)

    print('We have a total of ' + str(len(image_filenames)) + ' images')
    print(f'Used {len(train_filenames)} train images')
    print(f'Used {len(test_filenames)} test images')
    print('Categories names saved')

if __name__ == "__main__":
    main()
