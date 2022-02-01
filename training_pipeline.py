import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import KFold
from PIL import Image
import random

import skimage as sk
from skimage import transform
from skimage import util
from skimage import io

from augmentations import *


gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)


####Paramaters for data augmentation####

#Value that determines the chance that a pixel in an image is changed to noise
var = 0.075

num_rot = 2 # number of rotated images to augment from each original image
num_noise = 2 # number of noise image augmented from each original image

#Value that determines by how much the data augmentation method should create new data
data_multiplier = num_rot + num_noise

####Parameters for the CNN####
#Activation function
activation = 'relu'
#Optimization algorithm
optimizer = 'adam'

data = []
#load thedata
with open('mnist_digits_data.txt') as f:
  lines = f.readlines()
  # print(len(lines))
  for line in lines:
    
    line = line.split('  ')
    line.pop(0)
    line[-1] = line[-1].replace("\n", "")
    for idx in range(len(line)):
      line[idx] = int(line[idx])

    # print(len(line))
    data.append(line)

data = np.asarray(data)


#Create a second set of labels for the original dataset
labels_original_data = np.zeros(200)
for i in range (1, 10):
  labels_original_data = np.concatenate((labels_original_data, np.zeros(200)+i))


# print(labels)
#Normalize the pixel values between 0-1
data = data / np.amax(data)


#Create data to be multidimensional
data = np.reshape(data, (2000, 16, 15, 1))

print("Shape of original 2d data = " + str(data.shape))

data_augmented = []

for image in data:
    for i in range(num_rot):
        rot_img = random_rotation(image)
        data_augmented.append(rot_img)
    for j in range(num_noise):
        noise_img = random_noise(image, var)
        data_augmented.append(noise_img)

plt.imshow(data_augmented[3], cmap = 'Greys', interpolation='nearest')
plt.show()

data_augmented = np.asarray(data_augmented)
print("Shape of augmented 2d data = " + str(data_augmented.shape))

#labels for augmented data
#labels are 0-9, 200 of each in the base dataset
labels_aug = np.zeros(200 * data_multiplier)
for i in range (1, 10):
  labels_aug = np.concatenate((labels_aug, np.zeros(200 * data_multiplier)+i))


#Define the Kfold Cross Validation Model
folds = 10
kfold = KFold(n_splits = folds, shuffle = True)

accuracy_per_fold = []
loss_per_fold = []
model_per_fold = []

current_fold = 1
for train, test in kfold.split(data_augmented, labels_aug):


  #gepakt van een tutorial, moeten we aanpassen
  model = keras.Sequential([
      keras.layers.Conv2D(filters = 64, kernel_size = (3,3), activation = activation, input_shape = (16, 15, 1)),
      keras.layers.MaxPooling2D((2,2)),
      keras.layers.Flatten(),
      keras.layers.Dense(64, activation= activation),
      keras.layers.Dense(10, activation='softmax')
      #add layers
  ])
  model.compile(
      optimizer= optimizer,
      loss='sparse_categorical_crossentropy',
      metrics=['accuracy']
  )

  print("###################################")
  print("Fold Number:" + str(current_fold))

  print(data_augmented[train].shape)
  print(labels_aug[train].shape)
  history = model.fit(data_augmented[train], labels_aug[train], epochs = 10)
  model_per_fold.append(model)
  score = model.evaluate(data_augmented[test], labels_aug[test], verbose = 0)

  accuracy_per_fold.append(score[1] * 100)
  loss_per_fold.append(score[0])
  current_fold += 1

for idx in range(current_fold - 1):
  print("Accuracy for fold: " + str(idx + 1)+ " = " + str(accuracy_per_fold[idx]))
  print("Loss for fold: " + str(idx + 1)+ " = " + str(loss_per_fold[idx]))



#take the model with the highest accuracy from the cross validation
best_model = model_per_fold[accuracy_per_fold.index(max(accuracy_per_fold))]

#shuffle the original data and its labels
data, labels_original_data = shuffle(data, labels_original_data, random_state = 0)

#evaluate the model best on the original data
score_original_data = model.evaluate(data, labels_original_data, verbose = 0)
print("################################################")
print("Accuracy on original data = " + str(score_original_data[1] * 100))
print("Loss on original data = " + str(score_original_data[0]))








