#!/usr/bin/env python3
# Sistemas Avançados de Visão Industrial (SAVI 22-23)
# José Nuno Cunha, DEM, UA


# from copy import deepcopy
# from functools import partial
# from random import randint
# from matplotlib import cm
# from more_itertools import locate

import cv2
import numpy as np
import os
import imutils
import time
import copy

from colorama import Fore, Style
from datetime import date, datetime
from matplotlib import pyplot as plt

# Global variables
today = date.today()
today_date = today.strftime("%B %d, %Y")


def main():

    print(Fore.RED + "SAVI:", Style.RESET_ALL + "Get the images classes, " + today_date)


    # --------------------------------------
    # Initialization
    # --------------------------------------

    # get the dataset images to teach the program
    dataset_dir = 'rgbd-dataset/teach_dataset'

    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    

    # List all files in the directory
    files = os.listdir(dataset_dir)

    # Save the classes of the objects in a list
    classes = []

    # Print the names of the files
    for file in files:
        print(file)

        # get all the classes names from the objects file names
        class_name = file.split("_")[0]

        # add those names to the list "classes"
        if class_name not in classes:
            classes.append(class_name)

        # construct the full path of the image
        img_path = os.path.join(dataset_dir, file)

        # read the image
        img = cv2.imread(img_path)

        # work with the image gui
        img_gui = copy.deepcopy(img)

        img_gray = cv2.cvtColor(img_gui, cv2.COLOR_BGR2GRAY)
        img_rgb = cv2.cvtColor(img_gui, cv2.COLOR_BGR2RGB)



        # --------------------------------------
        # Visualization
        # --------------------------------------
        cv2.imshow('Objects', img_gui)
        

        # Wait for 3 seconds (3000 milliseconds)
        k = cv2.waitKey(3000) & 0xFF

        # press 'n' to see the next picture
        if k == ord('n'):
            continue

        elif k == ord('q'):
            break

    

    print(classes)

    # save the classes names in a txt file
    txt_classes = 'classes.txt'
    with open(txt_classes, 'w') as file:
        for class_name in classes:
            file.write(class_name + '\n')
    
    print(f"Classes saved to {txt_classes}")



    # --------------------------------------
    # Termination
    # --------------------------------------

        
    cv2.destroyAllWindows()



    


if __name__ == "__main__":
    main()