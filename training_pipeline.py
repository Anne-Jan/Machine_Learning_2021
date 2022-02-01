import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import KFold
from PIL import Image
import random

from sklearn.model_selection import train_test_split
import skimage as sk
from skimage import transform, util, io


from augmentations import *


gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)


####Paramaters for data augmentation####

#Value that determines the chance that a pixel in an image is changed to noise
var = 0.075

num_rot = 1 # number of rotated images to augment from each original image
num_noise = 1 # number of noise image augmented from each original image

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


#Create a split, use one for the training pipeline, with data augmentation.
train_original_data, validation_original_data, train_labels_original_data, validation_labels_original_data = train_test_split(data, labels_original_data, test_size=0.2, random_state=0)



data_augmented = []
labels_aug = []

#Idx used to get the correct label of the image we want to create augmentations of
labelidx = 0
for image in train_original_data:
  label = train_labels_original_data[labelidx]
  for i in range(num_rot):
    rot_img = random_rotation(image)
    data_augmented.append(rot_img)
    labels_aug.append(label)
  for j in range(num_noise):
    noise_img = random_noise(image, var)
    data_augmented.append(noise_img)
    labels_aug.append(label)
  labelidx += 1

labels_aug = np.asarray(labels_aug)
# plt.imshow(data_augmented[3], cmap = 'Greys', interpolation='nearest')
# plt.show()
# print(labels_aug[3])

data_augmented = np.asarray(data_augmented)
print("Shape of augmented 2d data = " + str(data_augmented.shape))


# Split augmented data in train and test data
X_train, X_test, y_train, y_test = train_test_split(data_augmented, labels_aug, test_size=0.2, random_state=0)

#Define the Kfold Cross Validation Model
folds = 10
kfold = KFold(n_splits = folds, shuffle = True)

accuracy_per_fold = []
loss_per_fold = []
model_per_fold = []

current_fold = 1
#Perform k-fold cross validation on the train split
for train, test in kfold.split(X_train, y_train):

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

  history = model.fit(X_train[train], y_train[train], epochs = 10, validation_data = (X_train[test], y_train[test]))
  model_per_fold.append((model, history))
  score = model.evaluate(X_train[test], y_train[test], verbose = 0)

  accuracy_per_fold.append(score[1] * 100)
  loss_per_fold.append(score[0])
  current_fold += 1

for idx in range(current_fold - 1):
  print("Accuracy for fold: " + str(idx + 1)+ " = " + str(accuracy_per_fold[idx]))
  print("Loss for fold: " + str(idx + 1)+ " = " + str(loss_per_fold[idx]))


#Take the model with the highest accuracy from the cross validation
best_model, best_history = model_per_fold[accuracy_per_fold.index(max(accuracy_per_fold))]

#Plot the accuracy versus the validation accuracy

plt.plot(best_history.history['accuracy'], label='accuracy')
plt.plot(best_history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()



#Shuffle the validation set of the original data and its labels
data, labels_original_data = shuffle(validation_original_data, validation_labels_original_data, random_state = 0)

#evaluate the model best on the augmented version of the test data created from the original data
score_original_data = model.evaluate(validation_original_data, validation_labels_original_data, verbose = 0)
label_probabilities = model.predict(validation_original_data)
predicted_labels = tf.argmax(label_probabilities, axis = 1)
print(predicted_labels)
print(predicted_labels.shape)
print(tf.math.confusion_matrix(labels = validation_labels_original_data, predictions = predicted_labels, num_classes = 10))

print("################################################")
print("Accuracy on original data = " + str(score_original_data[1] * 100))
print("Loss on original data = " + str(score_original_data[0]))


#evaluate the model best on the augmented test data
score_aug_test = model.evaluate(X_test, y_test, verbose = 0)

print("################################################")
print("Accuracy on augmented test data = " + str(score_aug_test[1] * 100))
print("Loss on augmented test data = " + str(score_aug_test[0]))



