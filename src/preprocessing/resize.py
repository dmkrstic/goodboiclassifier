from PIL import Image
from resizeimage import resizeimage
from os import listdir

folder = 'C:\\Users\\Dusica Krstic\\Documents\\GitHub\\goodboiclassifier\\dataset\\training_set'
folder_list = listdir(folder)
    
for folder_url in folder_list:
    breed_list = listdir(folder + '\\' + folder_url)
    for breed_url in breed_list:
        img_list = listdir(folder + '\\' + folder_url + '\\' + breed_url)
        for img_url in img_list:
            if img_url.endswith('.jpg') or img_url.endswith('.png') or img_url.endswith('.jpeg'):
                try:
                    fd_img = open(folder + '\\' + folder_url + '\\' + breed_url + '\\' + img_url, 'rb')
                    img = Image.open(fd_img)
                    img = resizeimage.resize_cover(img, [144, 144])
                    img.save(folder + '\\' + folder_url + '\\' + breed_url + '\\' + img_url, img.format)
                    fd_img.close()
                except:
                    print("error: " + folder + '\\' + folder_url + '\\' + breed_url + '\\' + img_url)
                    continue
                