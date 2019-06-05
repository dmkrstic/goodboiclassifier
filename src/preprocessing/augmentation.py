# Creates "new" images from existing images
import keras
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import imageio
import matplotlib.pyplot as plt
from PIL import Image
from os import listdir
import os

transformation = ImageDataGenerator(rotation_range = 359, width_shift_range = -0.1, height_shift_range = -0.1,
                                    zoom_range = 0.1, fill_mode = 'nearest', horizontal_flip = True,
                                    vertical_flip = True)

folder = 'C:\\Users\\Dusica Krstic\\Documents\\GitHub\\goodboiclassifier\\dataset\\training_set\\09-Sighthounds'
breed_list = os.listdir(folder)

for breed_url in breed_list:
    img_list = listdir(folder + '\\' + breed_url)
    for img_url in img_list:
            img = np.expand_dims(np.expand_dims(imageio.imread(folder + '\\' + breed_url + '\\' + img_url), 0), 3)
            try: 
                img_new = transformation.flow(img, y = None)
                img_augmented = [next(img_new)[0].astype(np.uint8) for i in range(2)]
                for i in range(2):
                        if img_url.endswith('.jpg'):
                            imageio.imwrite(folder + '\\' + breed_url + '\\' + img_url[:-4] + str(i) + '.jpg', img_augmented[i]) 
                        if img_url.endswith('.jpeg'):
                            imageio.imwrite(folder + '\\' + breed_url + '\\' + img_url[:-4] + str(i) + '.jpeg', img_augmented[i]) 
                        if img_url.endswith('.png'):
                            imageio.imwrite(folder + '\\' + breed_url + '\\' + img_url[:-4] + str(i) + '.png', img_augmented[i]) 
            except:
                print(img_url)
                continue