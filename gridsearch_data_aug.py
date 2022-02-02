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


averages = []
nruns = 10
for idx in range(0, nruns):
  print("Currently on run: " + str(idx))
  results_gridsearch = []
  parameters = []
  for var in [0.02, 0.04, 0.06, 0.08, 0.1]:
    for data_multiplier in [2, 4, 6, 8, 10]:
      #Store the parameters used for this run
      parameters.append("Var = " + str(var) + " num of aug = " + str(data_multiplier))

      ####Parameters for the CNN####
      #Activation function
      activation = 'relu'
      #Optimization algorithm
      optimizer = 'adam'

      ### Load data ###
      data = []
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


      #Create labels for the original dataset
      labels_original_data = np.zeros(200)
      for i in range (1, 10):
        labels_original_data = np.concatenate((labels_original_data, np.zeros(200)+i))


      # print(labels)
      #Normalize the pixel values between 0-1
      data = data / np.amax(data)


      #Create data to be multidimensional
      data = np.reshape(data, (2000, 16, 15, 1))



      #Create a split, use one for the training pipeline, with data augmentation.
      train_original_data, validation_original_data, train_labels_original_data, validation_labels_original_data = train_test_split(data, labels_original_data, test_size = 0.2, stratify = labels_original_data, random_state=0)


      ### Below data augmentation is done on the original training data and validation data. 

      ### Augmentation on train data
      data_augmented_train = []
      labels_aug_train = []

      #Idx used to get the correct label of the image we want to create augmentations of
      labelidx = 0
      for image in train_original_data:
        label = train_labels_original_data[labelidx]
        for i in range(data_multiplier):
          rot_img = random_rotation(image)
          noise_and_rot_img = random_noise(rot_img, var)
          data_augmented_train.append(noise_and_rot_img)
          labels_aug_train.append(label)
        labelidx += 1

      labels_aug_train = np.asarray(labels_aug_train)
      # plt.imshow(data_augmented[3], cmap = 'Greys', interpolation='nearest')
      # plt.show()

      ### Augmentation on validation data
      data_augmented_test = []
      labels_aug_test = []

      labelidx = 0
      for image in validation_original_data:
        label = validation_labels_original_data[labelidx]
        for i in range(data_multiplier):
          rot_img = random_rotation(image)
          noise_and_rot_img = random_noise(rot_img, var)
          data_augmented_test.append(noise_and_rot_img)
          labels_aug_test.append(label)
        labelidx += 1

      labels_aug_test = np.asarray(labels_aug_test)

      data_augmented_train = np.asarray(data_augmented_train)
      data_augmented_test = np.asarray(data_augmented_test)

      #Define the Kfold Cross Validation Model
      folds = 10
      kfold = KFold(n_splits = folds, shuffle = True)

      accuracy_per_fold = []
      loss_per_fold = []
      model_per_fold = []

      current_fold = 1
      #Perform k-fold cross validation on the train split
      for train, test in kfold.split(data_augmented_train, labels_aug_train):
        
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
        # early_stopping = keras.callbacks.EarlyStopping(monitor = 'val_loss', mode = 'auto', verbose = 0)
        # history = model.fit(data_augmented_train[train], labels_aug_train[train], epochs = 20, validation_data = (data_augmented_train[test], labels_aug_train[test]), callbacks = [early_stopping])
        history = model.fit(data_augmented_train[train], labels_aug_train[train], epochs = 5, validation_data = (data_augmented_train[test], labels_aug_train[test]), verbose = 0)
        model_per_fold.append((model, history))
        score = model.evaluate(data_augmented_train[test], labels_aug_train[test], verbose = 0)

        accuracy_per_fold.append(score[1] * 100)
        loss_per_fold.append(score[0])
        current_fold += 1


      #Take the model with the highest accuracy from the cross validation
      best_model, best_history = model_per_fold[accuracy_per_fold.index(max(accuracy_per_fold))]
      results_gridsearch.append(((max(best_history.history['val_accuracy']))))


  
  averages.append(results_gridsearch)

#Average the results over 10 runs
averaged_gridsearch_results = []
for result_idx in range(len(results_gridsearch)):
  averaged_result = 0
  for idx in range(0, nruns):
    averaged_result += averages[idx][result_idx]
    hyperparams = averages[idx][result_idx]
  averaged_result = averaged_result / nruns
  averaged_gridsearch_results.append((averaged_result, parameters[result_idx]))

for idx in range(len(averaged_gridsearch_results)):
  print(averaged_gridsearch_results[idx])



  

      # plt.plot(best_history.history['accuracy'], label='accuracy')
      # plt.plot(best_history.history['val_accuracy'], label = 'val_accuracy')
      # plt.xlabel('Epoch')
      # plt.ylabel('Accuracy')
      # plt.ylim([0.5, 1])
      # plt.legend(loc='lower right')
      # plt.show()


