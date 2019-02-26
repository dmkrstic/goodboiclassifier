from PIL import Image
from resizeimage import resizeimage
import os

folder = 'C:\\Users\\NZivkovic\\Documents\\GitHub\\goodboiclassifier\\Training_set'
folder_list = os.listdir(folder)
    
for folder_url in folder_list:
    breed_list = os.listdir(folder + '\\' + folder_url)
    for breed_url in breed_list:
        img_list = os.listdir(folder + '\\' + folder_url + '\\' + breed_url)
        i = 1
        for img_url in img_list:
            if img_url.endswith('.jpg') or img_url.endswith('.png') or img_url.endswith('.jpeg'):
                path = folder + '\\' + folder_url + '\\' + breed_url + '\\' + img_url
                newpath = folder + '\\' + folder_url + '\\' + breed_url + '\\' + str(i) + '_' + breed_url
                try:
                    if img_url.endswith('.jpg'):
                        os.rename(path, newpath + '.jpg')
                    if img_url.endswith('.png'):
                        os.rename(path, newpath + '.png')
                    if img_url.endswith('.jpeg'):
                        os.rename(path, newpath + '.jpeg')
                except:
                    print("error")
                    continue
            i = i + 1 