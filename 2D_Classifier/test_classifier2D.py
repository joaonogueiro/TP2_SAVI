#!/usr/bin/env python3

import cv2
from PIL import Image
from classifier_2D import Classifier_2D

def main():

    image = cv2.imread('../2D_Classifier/Images test/Mug.png')
    # Será necessário ajustar a rede...???

    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # -----------------------------------------------------------------
    # Predicted
    # -----------------------------------------------------------------
    classifier = Classifier_2D(image_input=image_pil)
    prediction = classifier.predicted()
    print(f'Predicted label: {prediction}')

if __name__ == "__main__":
    main()
