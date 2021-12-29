import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow import keras


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
print(data[0].shape)
#labels are 0-9
labels = np.zeros(200)
for i in range (1, 10):
  labels = np.concatenate((labels, np.zeros(200)+i))


#shuffle data and labels

data, labels = shuffle(data, labels, random_state = 0)

#create the train/test split 80/20
trainX, trainY = data[:1600], labels[:1600]
testX, testY = data[1600:], labels[1600:]


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

history = model.fit(trainX, trainY, epochs = 10)
model.evaluate(testX, testY)








