import numpy as np 
import keras
from keras import backend as K 
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense, Flatten, Dropout, Activation
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from matplotlib import pyplot as plt 
from sklearn.metrics import confusion_matrix
import itertools
from keras.models import load_model

dataset_folder = '/content/dataset/'

training_set = dataset_folder + 'training_set'
valid_set = dataset_folder + 'valid_set'
test_set = dataset_folder + 'test_set'

class_list = ['00-Caniche', '01-Deutscher Schaeferhund','02-Rottweiler','03-Schnauzer','04-Alaskan malamute',
              '05-Beagle', '06-Magyar vizsla','07-Golden retriever','08-Chihuahua','09-Picolo levriero italiano']

training_batch = ImageDataGenerator().flow_from_directory(training_set, target_size = (144, 144),
                    classes = class_list, batch_size = 32, color_mode = 'grayscale')
valid_batch = ImageDataGenerator().flow_from_directory(valid_set, target_size = (144, 144),
                    classes = class_list, batch_size = 32, color_mode ='grayscale')
test_batch = ImageDataGenerator().flow_from_directory(test_set, target_size = (144, 144),
                    classes = class_list, batch_size = 32, color_mode = 'grayscale', shuffle = False)


model = load_model('model.h5')

history = model.fit_generator(training_batch, steps_per_epoch = 1087, validation_data = valid_batch,
                    validation_steps = 30, epochs = 1, verbose = 2)

model.save('goodboiclassifier.h5')
