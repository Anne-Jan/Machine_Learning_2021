import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import KFold
from PIL import Image
import random


data = []
#Value that determines by how much the data augmentation method should create new data
data_multiplier = 5

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
#labels are 0-9, 200 of each in the base dataset
labels = np.zeros(200 * data_multiplier)
for i in range (1, 10):
  labels = np.concatenate((labels, np.zeros(200 * data_multiplier)+i))

print(labels)
#Normalize the pixel values between 0-1
data = data / np.amax(data)



# pic = data[0]
# for idx in range(len(pic)):
#   if(random.uniform(0, 1) < 0.2):
#     pic[idx] = random.uniform(0, 1)

# ###Show one datapoint for testing of data augmentation   
# pic = np.array(pic)
# picmatreverse = np.zeros((14,15))
# counter = 0
# for column in range(14):
#   for row in range(15):
#     picmatreverse[column][row] = picmatreverse[column][row] - pic[counter]
#     counter += 1
# picmat = np.zeros((14,15))
# for k in range(14):
#   picmat[:, k] = picmatreverse[:, 14 - k]
#   # print(picmat)
# plt.gray()
# plt.imshow(picmatreverse)
# plt.colorbar()
# plt.show()

new_data = []
for idx1 in range(len(data)):
  pic = data[idx1]
  #For each image, create multiple variations of the image by adding noise.
  #The amount of variations per image is determined by the data_multiplier variable
  for idx2 in range(data_multiplier):
    pic_to_augment = pic.copy()    
    for idx3 in range(len(pic_to_augment)):
      #For each pixel, small chance to add noise
      if(random.uniform(0, 1) < 0.15):
        pic_to_augment[idx3] = random.uniform(0, 1)
    #Add the newly generated digit to the augmented dataset    
    new_data.append(pic_to_augment)


data = np.asarray(new_data)
print(data.shape)

pic = data[0]

# ###Show one datapoint for testing of data augmentation   
# pic = np.array(pic)
# picmatreverse = np.zeros((14,15))
# counter = 0
# for column in range(14):
#   for row in range(15):
#     picmatreverse[column][row] = picmatreverse[column][row] - pic[counter]
#     counter += 1
# picmat = np.zeros((14,15))
# for k in range(14):
#   picmat[:, k] = picmatreverse[:, 14 - k]
#   # print(picmat)
# plt.gray()
# plt.imshow(picmatreverse)
# plt.colorbar()
# plt.show()


#Define the Kfold Cross Validation Model
folds = 10
kfold = KFold(n_splits = folds, shuffle = True)

accuracy_per_fold = []
loss_per_fold = []

current_fold = 1
for train, test in kfold.split(data, labels):


  #gepakt van een tutorial, moeten we aanpassen
  model = keras.Sequential([
      keras.layers.Dense(10, input_shape=(240,), activation='sigmoid')
      #add layers
  ])
  model.compile(
      optimizer='adam',
      loss='sparse_categorical_crossentropy',
      metrics=['accuracy']
  )

  print("###################################")
  print("Fold Number:" + str(current_fold))

  history = model.fit(data[train], labels[train], epochs = 10)
  score = model.evaluate(data[test], labels[test], verbose = 0)

  accuracy_per_fold.append(score[1] * 100)
  loss_per_fold.append(score[0])
  current_fold += 1

for idx in range(current_fold - 1):
  print("Accuracy for fold: " + str(idx + 1)+ " = " + str(accuracy_per_fold[idx]))
  print("Loss for fold: " + str(idx + 1)+ " = " + str(loss_per_fold[idx]))







