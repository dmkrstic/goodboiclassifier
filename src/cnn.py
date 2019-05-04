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

#dataset_folder = 'C:\\Users\\Dusica Krstic\\Documents\\GitHub\\goodboiclassifier\\dataset'
dataset_folder = '/content/dataset/'

training_set = dataset_folder + 'training_set'
valid_set = dataset_folder + 'valid_set'
test_set = dataset_folder + 'test_set'

class_list = ['01-Sheepdogs and Cattledogs','02-Pinscher and Schnauzer','03-Terriers','04-Spitz',
              '05-Scent hounds', '06-Pointing Dogs','07-Retrievers','08-Companion and Toy dogs','09-Sighthounds']

#remove color_mode for RGB images
training_batch = ImageDataGenerator().flow_from_directory(training_set, target_size = (144, 144),
                    classes = class_list, batch_size = 32, color_mode = 'grayscale')
valid_batch = ImageDataGenerator().flow_from_directory(valid_set, target_size = (144, 144),
                    classes = class_list, batch_size = 32, color_mode ='grayscale')
test_batch = ImageDataGenerator().flow_from_directory(test_set, target_size = (144, 144),
                    classes = class_list, batch_size = 32, color_mode = 'grayscale', shuffle = False)


model = Sequential()

#change input shape to (144, 144, 3) for RGB images
model.add(Conv2D(64, (3, 3), padding = 'same', input_shape = (144, 144, 1)))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3), padding = 'same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(128, (3, 3), padding = 'same'))
model.add(Activation('relu'))
model.add(Conv2D(128, (3, 3), padding = 'same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.1))

model.add(Conv2D(256, (3, 3), padding = 'same'))
model.add(Activation('relu'))
model.add(Conv2D(256, (3, 3), padding = 'same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.15))

model.add(Conv2D(512, (3, 3), padding = 'same'))
model.add(Activation('relu'))
model.add(Conv2D(512, (3, 3), padding = 'same'))
model.add(Activation('relu'))
model.add(Conv2D(512, (3, 3), padding = 'same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.15))

model.add(Flatten())

model.add(Dense(1536))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2048))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(9))
model.add(Activation('softmax'))

opt = keras.optimizers.Adam(lr = 0.0001)

model.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics = ['accuracy'])

model.fit_generator(training_batch, steps_per_epoch = 280, validation_data = valid_batch,
                    validation_steps = 30, epochs = 30, verbose = 2)

model.save('goodboiclassifier.h5')

#test_img, test_label = next(test_batch)
#for i in range(64):
#    test_img1, test_label1 = next(test_batch)
#    test_label = np.vstack((test_label, test_label1))

#predictions = model.predict_generator(test_batch, steps = 65, verbose = 2)

#test_label = test_label[:, 0]
#for i in predictions:
#	if i[0] >= i[1]:
#		i[0] = 1
#	else:
#		i[0] = 0
#cm = confusion_matrix(test_label, predictions[:, 0])

#funkcija copy-paste-ovana sa scikit-learn.org
#def plot_confusion_matrix(cm, classes,
#                          normalize = False,
#                          title = 'Confusion matrix'):
#    """
#    This function prints and plots the confusion matrix.
#    Normalization can be applied by setting `normalize=True`.
#    """
#    if normalize:
#        cm = cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis]
#        print("Normalized confusion matrix")
#    else:
#        print('Confusion matrix, without normalization')
#
#    print(cm)
#
#    plt.imshow(cm, interpolation = 'nearest')
#    plt.title(title)
#    plt.colorbar()
#    tick_marks = np.arange(len(classes))
#    plt.xticks(tick_marks, classes, rotation = 45)
#    plt.yticks(tick_marks, classes)
#
#    fmt = '.2f' if normalize else 'd'
#    thresh = cm.max() / 2.
#    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#        plt.text(j, i, format(cm[i, j], fmt),
#                 horizontalalignment = "center",
#                 color = "white" if cm[i, j] > thresh else "black")
#
#    plt.ylabel('True label')
#    plt.xlabel('Predicted label')
#    plt.tight_layout()
#    plt.show()
#
#
#cm_plot_labels = class_list
#plot_confusion_matrix(cm, cm_plot_labels, title = 'Confusion matrix')
