#!/usr/bin/env python3

import os
import glob
import json
from sklearn.model_selection import train_test_split

# This program ensures that the images that are used for the 
#various processes (training, validation and testing) are not repeated

def categoriesNames(dataset_path):
    # Save the name of the categories by the name of the folders
    if os.path.isdir(dataset_path):
        labels_list = [name for name in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, name))]
    else:
        print(f'ERROR The path not exist: ({dataset_path}).')

    return labels_list
       
def main():

    dataset_path = '../../rgbd-dataset' 
    image_filenames = glob.glob(dataset_path + '/*/*/*_crop.png')

    # Check if dataset data exists
    if len(image_filenames) < 1:
        raise FileNotFoundError('Dataset files not found')
    
    # Splitting the dataset
    train_filenames, remaining_filenames = train_test_split(image_filenames, test_size=0.3)
    validation_filenames, test_filenames = train_test_split(remaining_filenames, test_size=0.33)

    print('We have a total of ' + str(len(image_filenames)) + ' images.')
    print(f'Used {len(train_filenames)} train images')
    print(f'Used {len(validation_filenames)} validation images')
    print(f'Used {len(test_filenames)} test images')

    labels_list = categoriesNames(dataset_path)

    d = {'train_filenames': train_filenames,
         'validation_filenames': validation_filenames,
         'test_filenames': test_filenames,
         'labels_names': labels_list}

    json_object = json.dumps(d, indent=2)

    # Writing to sample.json
    with open("dataset_filenames.json", "w") as outfile:
        outfile.write(json_object)

if __name__ == "__main__":
    main()
