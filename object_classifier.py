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

    print(Fore.GREEN + "SAVI:", Style.RESET_ALL + "Detect the objects in the images, " + today_date)


    # --------------------------------------
    # Initialization
    # --------------------------------------

    # import the classes of the images
    classes = []
    with open("classes.txt", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    
    print(classes)




    # ------------------- Load the images -------------------
    # get the dataset images to teach the program
    dataset_dir = 'rgbd-dataset/teach_dataset'

    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    

    # List all files in the directory
    files = os.listdir(dataset_dir)

    for file in files:
        # print(file)

        img_path = os.path.join(dataset_dir, file)

        # read the image
        img = cv2.imread(img_path)

        # work with the image gui
        img_gui = copy.deepcopy(img)
        # img_gui = cv2.resize(img_gui, None, fx=0.4, fy=0.4)
        print("Original Dimensions: ", img_gui.shape)

        h, w, _ = img_gui.shape

        cv2.rectangle(img_gui, (0, 0), (w, h), (0, 255, 0), 2)


        # TODO: agr comparar a imagem que está dentro do retângulo verde com as imagens modelo de cada class
        # TODO: medir a accuracy/confidence e atribuir a class correta ao ficheiro, tipo, condifence >= 0.85
        





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
    

    # --------------------------------------
    # Termination
    # --------------------------------------

        
    cv2.destroyAllWindows()



    
if __name__ == "__main__":
    main()