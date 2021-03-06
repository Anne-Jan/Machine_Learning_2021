import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import KFold
from PIL import Image
import random
import seaborn as sns

from sklearn.model_selection import train_test_split
import skimage as sk
from skimage import transform, util, io

from augmentations import *


gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)


####Paramaters for data augmentation####

# variance for gaussian distributed noise
var = 0.02

num_aug = 10 # number of created augmented images (combination of rotation and noise) per original image

#Value that determines by how much the data augmentation method should create new data
data_multiplier = num_aug

####Parameters for the CNN####
#Activation function
activation = 'tanh'
#Optimization algorithm
optimizer = 'adam'

### Load data ###
data = []
with open('mfeat-pix.txt') as f:
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


#Create labels for the original dataset
labels_original_data = np.zeros(200)
for i in range (1, 10):
  labels_original_data = np.concatenate((labels_original_data, np.zeros(200)+i))


#Normalize the pixel values between 0-1
data = data / np.amax(data)


#Create data to be multidimensional
data = np.reshape(data, (2000, 16, 15, 1))


#Create a stratified split, use one for the training pipeline, with data augmentation.
train_original_data, validation_original_data, train_labels_original_data, validation_labels_original_data = train_test_split(data, labels_original_data, test_size = 0.2, stratify = labels_original_data, random_state=0)


### Below data augmentation is done on the original training data and validation data. 

### Augmentation on train data
data_augmented_train = []
labels_aug_train = []

#Idx used to get the correct label of the image we want to create augmentations of
labelidx = 0
for image in train_original_data:
  label = train_labels_original_data[labelidx]
  for i in range(num_aug):
    rot_img = random_rotation(image)
    noise_and_rot_img = random_noise(rot_img, var)
    data_augmented_train.append(noise_and_rot_img)
    labels_aug_train.append(label)
  labelidx += 1

#Convert to numpy arrays for the cnn
labels_aug_train = np.asarray(labels_aug_train)

### Augmentation on validation data
data_augmented_test = []
labels_aug_test = []

labelidx = 0
for image in validation_original_data:
  label = validation_labels_original_data[labelidx]
  for i in range(num_aug):
    rot_img = random_rotation(image)
    noise_and_rot_img = random_noise(rot_img, var)
    data_augmented_test.append(noise_and_rot_img)
    labels_aug_test.append(label)
  labelidx += 1

#Convert to numpy arrays for the cnn
labels_aug_test = np.asarray(labels_aug_test)
data_augmented_train = np.asarray(data_augmented_train)
data_augmented_test = np.asarray(data_augmented_test)



#Define the Kfold Cross Validation Model
folds = 10
kfold = KFold(n_splits = folds, shuffle = True)

accuracy_per_fold = []
model_per_fold = []
results_per_fold = []

current_fold = 1
#Perform k-fold cross validation on the train split
for train, test in kfold.split(data_augmented_train, labels_aug_train):
  
  model = keras.Sequential([
      keras.layers.Conv2D(filters = 64, kernel_size = (3,3), activation = activation, input_shape = (16, 15, 1)),
      keras.layers.MaxPooling2D((2,2)),
      keras.layers.Dropout(0.2),
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

  history = model.fit(data_augmented_train[train], labels_aug_train[train], epochs = 20, validation_data = (data_augmented_train[test], labels_aug_train[test]))
  model_per_fold.append((model, history))

  #Append the val accuracy of the model of this fold
  results_per_fold.append(np.mean(history.history['val_accuracy']))
  score = model.evaluate(data_augmented_train[test], labels_aug_train[test], verbose = 0)

  accuracy_per_fold.append(np.mean(history.history['val_accuracy']) * 100)
  current_fold += 1


#Take the model with the highest val accuracy from the cross validation
best_model, best_history = model_per_fold[accuracy_per_fold.index(max(accuracy_per_fold))]

print("Averaged val accuracy of the kfold cross validation = " + str(sum(results_per_fold) / len(results_per_fold)))


#Plot the accuracy and val accuracy
plt.plot(best_history.history['accuracy'], label='accuracy')
plt.plot(best_history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.8, 1.05])
plt.legend(loc='lower right')
plt.grid()
plt.show()



####USE LATER AFTER TRAINING
#evaluate the model best on the test data that was held back
score_original_data = model.evaluate(validation_original_data, validation_labels_original_data, verbose = 0)

#Also create a confusion mat
label_probabilities = model.predict(validation_original_data)
predicted_labels = tf.argmax(label_probabilities, axis = 1)
cf_mat = tf.math.confusion_matrix(labels = validation_labels_original_data, predictions = predicted_labels, num_classes = 10)

sns.heatmap(cf_mat, annot = True)
plt.xlabel('Predicted Digit')
plt.ylabel('Actual Digit')
plt.show()

print("################################################")
print("Accuracy on original data = " + str(score_original_data[1] * 100))
print("Loss on original data = " + str(score_original_data[0]))

