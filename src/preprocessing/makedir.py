# Creates training, valid and test set folders
import os

tr_folder = 'C:\\Users\\Dusica Krstic\\Documents\\GitHub\\goodboiclassifier\\dataset\\training_set'
v_folder = 'C:\\Users\\Dusica Krstic\\Documents\\GitHub\\goodboiclassifier\\dataset\\valid_set'
t_folder = 'C:\\Users\\Dusica Krstic\\Documents\\GitHub\\goodboiclassifier\\dataset\\test_set'

t_groups = os.listdir(tr_folder)

for group in t_groups:
    t_breeds = os.listdir(tr_folder + '\\' + group)
    os.mkdir(v_folder + '\\' + group)
    os.mkdir(t_folder + '\\' + group)
    for breed in t_breeds:
        os.mkdir(v_folder + '\\' + group + '\\' + breed)
        os.mkdir(t_folder + '\\' + group + '\\' + breed)
