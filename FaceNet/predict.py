from PIL import Image
import numpy as np
import torch
from facenet import Facenet

if __name__ == "__main__":

    model = Facenet()
    # image_1 = Image.open('img/3.jpg')
    # model.encoder(image_1)
    #
    # exit()

    while True:
        image_1 = input('Input image_1 filename:')
        try:
            image_1 = Image.open(image_1)
        except:
            print('Image_1 Open Error! Try again!')
            continue

        image_2 = input('Input image_2 filename:')
        try:
            image_2 = Image.open(image_2)
        except:
            print('Image_2 Open Error! Try again!')
            continue
        
        probability = model.detect_image(image_1,image_2)
        print(probability)

