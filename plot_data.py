import matplotlib.pyplot as plt
import numpy as np


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

for i in range(1):
  for j in range(1):
    pic = data[1100]

    # pic = data[997+(i) + j]
    pic = np.array(pic)
    picmatreverse = np.zeros((14,15))
    counter = 0
    for column in range(14):
      for row in range(15):
        picmatreverse[column][row] = picmatreverse[column][row] - pic[counter]
        counter += 1
    picmat = np.zeros((14,15))
    for k in range(14):
      picmat[:, k] = picmatreverse[:, 14 - k]
      # print(picmat)
  plt.gray()
  plt.imshow(picmatreverse)
  plt.colorbar()
  plt.show()



