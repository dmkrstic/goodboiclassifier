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
from sklearn.metrics import classification_report
import itertools
from keras.models import load_model

dataset_folder = '/content/dataset/'

train_data_path = dataset_folder + 'training_set'
test_data_path = dataset_folder + 'test_set'

class_list = ['00-Caniche', '01-Deutscher Schaeferhund','02-Rottweiler','03-Schnauzer','04-Alaskan malamute',
              '05-Beagle', '06-Magyar vizsla','07-Golden retriever','08-Chihuahua','09-Picolo levriero italiano']
img_rows = 144
img_cols = 144

batch_size = 32
num_of_train_samples = 34840
num_of_test_samples = 6176

#Image Generator
train_generator = ImageDataGenerator().flow_from_directory(train_data_path,
                                                    target_size=(img_rows, img_cols),
                                                    batch_size=batch_size, color_mode='grayscale',
                                                    class_mode='categorical')

validation_generator = ImageDataGenerator().flow_from_directory(test_data_path,
                                                        target_size=(img_rows, img_cols),
                                                        batch_size=batch_size,
                                                        color_mode='grayscale',
                                                        shuffle=False,
                                                        class_mode='categorical')

model = load_model('model.h5')

#Confusion Matrix and Classification Report
Y_pred = model.predict_generator(validation_generator, num_of_test_samples // batch_size)
y_pred = np.argmax(Y_pred, axis=1)

print(validation_generator.classes-y_pred)

print('Confusion Matrix')
print(confusion_matrix(validation_generator.classes, y_pred))
print('Classification Report')
print(classification_report(validation_generator.classes, y_pred, target_names=class_list))