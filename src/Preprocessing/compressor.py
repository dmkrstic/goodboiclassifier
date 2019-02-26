import os
from os import listdir
from PIL import Image

folder = 'C:\\Users\\NZivkovic\\Documents\\GitHub\\goodboiclassifier\\Training_set'
folder_list = listdir(folder)
for folder_url in folder_list:
    breed_list = listdir(folder + '\\' + folder_url)
    for breed_url in breed_list:
        img_list = listdir(folder + '\\' + folder_url + '\\' + breed_url)
        for img_url in img_list:
            if img_url.endswith('.jpg') or img_url.endswith('.png') or img_url.endswith('.jpeg'):
                try:
                    info = os.stat(folder + '\\' + folder_url + '\\' + breed_url + '\\' + img_url)
                    if info.st_size > 1000000 and info.st_size <= 5000000:
                        img = Image.open(folder + '\\' + folder_url + '\\' + breed_url + '\\' + img_url)
                        img.save(folder + '\\' + folder_url + '\\' + breed_url + '\\' + img_url, quality = 65, optimize=True)
                    elif info.st_size > 5000000 and info.st_size <= 10000000:
                        img = Image.open(folder + '\\' + folder_url + '\\' + breed_url + '\\' + img_url)
                        img.save(folder + '\\' + folder_url + '\\' + breed_url + '\\' + img_url, quality = 55, optimize=True)
                    elif info.st_size > 10000000 and info.st_size <= 15000000:
                        img = Image.open(folder + '\\' + folder_url + '\\' + breed_url + '\\' + img_url)
                        img.save(folder + '\\' + folder_url + '\\' + breed_url + '\\' + img_url, quality = 45, optimize=True)
                    elif info.st_size > 15000000:
                        img = Image.open(folder + '\\' + folder_url + '\\' + breed_url + '\\' + img_url)
                        img.save(folder + '\\' + folder_url + '\\' + breed_url + '\\' + img_url, quality = 35, optimize=True)
                    print("success" + folder + '\\' + folder_url + '\\' + breed_url + '\\' + img_url)
                except:
                    print("error: " + folder + '\\' + folder_url + '\\' + breed_url + '\\' + img_url)
                    continue