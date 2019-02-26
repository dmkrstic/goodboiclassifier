from PIL import Image
from os import listdir

folder = 'C:\\Users\\NZivkovic\\Documents\\GitHub\\goodboiclassifier\\Training_set'
folder_list = listdir(folder)
for folder_url in folder_list:
    breed_list = listdir(folder + '\\' + folder_url)
    for breed_url in breed_list:
        img_list = listdir(folder + '\\' + folder_url + '\\' + breed_url)
        for img_url in img_list:
            if img_url.endswith('.jpg') or img_url.endswith('.png') or img_url.endswith('.jpeg'):
                try:
                    img = Image.open(folder + '\\' + folder_url + '\\' + breed_url + '\\' + img_url).convert('L')
                    img.save(folder + '\\' + folder_url + '\\' + breed_url + '\\' + img_url)
                except:
                    print("error: " + folder + '\\' + folder_url + '\\' + breed_url + '\\' + img_url)
                    continue