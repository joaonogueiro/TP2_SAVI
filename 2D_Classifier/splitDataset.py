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

    #######################################################################################
    # All categories
    #######################################################################################
    # dataset_folders = '../../rgbd-dataset' 
    # image_filenames = glob.glob(dataset_folders + '/*/*/*_crop.png')

    # # Check if dataset data exists
    # if len(image_filenames) < 1:
    #     raise FileNotFoundError('Dataset files not found')
    
    # # Speed up
    # image_filenames = random.sample(image_filenames, k=2000)

    # Slipt the Database 80% - Train and 20% - Test 
    # train_filenames, test_filenames = train_test_split(image_filenames, test_size=0.2)

    # labels_list = categoriesNames(dataset_folders)
    
    #######################################################################################
    # Only the objects in scenes
    #######################################################################################
    image_path = glob.glob('../../rgbd-dataset/*/*/*_crop.png')
    categories_scenes = ['bowl', 'cap', 'cereal_box', 'coffee_mug', 'soda_can'] 
    dataset_folders = '../../rgbd-dataset'
    selected_folders = [folder.split('/')[-1]
                        for obj in categories_scenes
                        for folder in os.listdir(os.path.join(dataset_folders, obj))
                        if os.path.isdir(os.path.join(dataset_folders, obj, folder))]

    print(f'Only these objects are selected: {categories_scenes} \n\nThe objects: {selected_folders}\n\nNumber of objects: {len(selected_folders)}')

    ################ Slipt the Database 80% - Train and 20% - Test ################ 
    # train_folders, test_folders = train_test_split(selected_folders, test_size=0.2)
    # print(f'For train we have {len(train_folders)}\n{train_folders}\n')
    # print(f'For test we have {len(test_folders)}\n{test_folders}\n')

    # Better split manualy, because are only 6 objects for test (Ensuring they are all tested)
    test_folders = ['bowl_5', 'cap_2', 'cereal_box_5', 'coffee_mug_7', 'soda_can_3', 'soda_can_1']
    train_folders = [folder for folder in selected_folders if folder not in test_folders]

    train_filenames = [filename for filename in image_path if any(f'/{obj}/' in filename for obj in train_folders)]
    train_filenames, validation_filenames = train_test_split(train_filenames, test_size=0.2) #################################
    test_filenames = [filename for filename in image_path if any(f'/{obj}/' in filename for obj in test_folders)]

    classes = {labels_list: i for i, labels_list in enumerate(categories_scenes)}
    json_categories = json.dumps(classes, indent=2)
    with open("categories.json", "w") as outfile:
        outfile.write(json_categories)
    
    # d = {'train_filenames': random.sample(train_filenames, len(train_filenames)),
        #  'test_filenames': random.sample(test_filenames, len(test_filenames))}
    d = {'train_filenames': train_filenames,
         'validation_filenames': validation_filenames,
         'test_filenames': random.sample(test_filenames, len(test_filenames))}


    json_object = json.dumps(d, indent=2)

    # Writing to sample.json
    with open("dataset_filenames.json", "w") as outfile:
        outfile.write(json_object)

    print('We have a total of ' + str(len(image_path)) + ' images')
    print(f'Used {len(train_filenames)} train images')
    print(f'Used {len(validation_filenames)} validation images')
    print(f'Used {len(test_filenames)} test images')
    print('Categories names saved')

if __name__ == "__main__":
    main()
