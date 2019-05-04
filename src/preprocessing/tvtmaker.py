import os
from os import listdir
from shutil import copyfile
import random

folder = 'C:\\Users\\Dusica Krstic\\Documents\\GitHub\\goodboiclassifier\\dataset'
category_list = os.listdir(folder + '\\' + 'training_set')

for category in category_list:
    breed_list = os.listdir(folder + '\\' + 'training_set' + '\\' + category)
    for breed in breed_list:
        img_list = os.listdir(folder + '\\' + 'training_set' + '\\' + category + '\\' + breed)
        for img_url in img_list:
            src = folder + '\\' + 'training_set' + '\\' +category + '\\' + breed + '\\' + img_url
            r = random.randint(1, 100)
            if r > 70 and r <= 85:
                #dest = folder + '\\' + 'valid_set' + '\\' + category + '\\' + breed + '\\' + img_url
                #os.rename(src, dest)
                continue
            elif r > 85:
                dest = folder + '\\' + 'test_set' + '\\' + category + '\\' + breed + '\\' + img_url
                #copyfile(src, dest)
                os.rename(src, dest)
            


