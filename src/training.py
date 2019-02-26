import os
import skimage
import matplotlib.pyplot as plt 
import numpy as np
import tensorflow as tf 
import keras as k 
import re
import random

def load_data(dir):
    dirs = [d for d in os.listdir(dir)]

    labels = []
    images = []

    for d in dirs:
        label_dir = os.path.join(dir, d)
        file_names = [os.path.join(label_dir, f)
                      for f in os.listdir(label_dir)
                      if f.endswith(".jpg")]
            
        for f in file_names:
            images.append(skimage.data.imread(f))
            d = re.search("[0-9]+", d).group()
            labels.append(int(d))
            print(f)

    return images, labels

#insert your path
#D: C:\\Users\Dusica Krstic\Documents\GitHub\goodboyclassifier
#N: C:\goodboyclassifier
ROOT_PATH = "C:\\Users\Dusica Krstic\Documents\GitHub\goodboyclassifier" 

train_data_directory = os.path.join(ROOT_PATH, "Temporary_training_set")

images, labels = load_data(train_data_directory)

dogs = [100, 300, 500, 700]

images56 = [skimage.transform.resize(image, (56, 56)) for image in images]

images56 = np.array(images56)

images56 = skimage.color.rgb2gray(images56)

#plot changed images
#for i in range(len(dogs)):
#    plt.subplot(1, 4, i+1)
#    plt.axis('off')
#    plt.imshow(images56[dogs[i]], cmap = "gray")
#    plt.subplots_adjust(wspace=0.5)

#plt.show()

x = tf.placeholder(dtype = tf.float32, shape = [None, 56, 56])

y = tf.placeholder(dtype = tf.int32, shape = [None])

images_flat = tf.contrib.layers.flatten(x)

logits = tf.contrib.layers.fully_connected(images_flat, 62, tf.nn.relu)

loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits = logits))

train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

correct_pred = tf.argmax(logits, 1)

accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

#print("images_flat: ", images_flat)
#print("logits: ", logits)
#print("loss: ", loss)
#print("predicted_labels: ", correct_pred)

tf.set_random_seed(1234)
sess = tf.Session()

sess.run(tf.global_variables_initializer())

#for i in range(201):
#        print('EPOCH', i)
#        _, accuracy_val = sess.run([train_op, accuracy], feed_dict={x: images56, y: labels})
#        if i % 10 == 0:
#            print("Loss: ", loss)
#        print('DONE WITH EPOCH')

sample_indexes = random.sample(range(len(images56)), 10)
sample_images = [images56[i] for i in sample_indexes]
sample_labels = [labels[i] for i in sample_indexes]

# Run the "correct_pred" operation
predicted = sess.run([correct_pred], feed_dict={x: sample_images})[0]
                        
# Print the real and predicted labels
print(sample_labels)
print(predicted)

# Display the predictions and the ground truth visually.
fig = plt.figure(figsize=(10, 10))
for i in range(len(sample_images)):
    truth = sample_labels[i]
    prediction = predicted[i]
    plt.subplot(5, 2,1+i)
    plt.axis('off')
    color='green' if truth == prediction else 'red'
    plt.text(40, 10, "Truth:        {0}\nPrediction: {1}".format(truth, prediction), 
             fontsize=12, color=color)
    plt.imshow(sample_images[i],  cmap="gray")

plt.show()