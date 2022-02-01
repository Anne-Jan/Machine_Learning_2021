import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from PIL import Image
import random
from scipy import ndarray

import skimage as sk
from skimage import transform
from skimage import util
from skimage import io

def random_rotation(image_array: ndarray):
    # pick a random degree of rotation between 45 degrees on the left and 45 degrees on the right
    random_degree = random.uniform(-45, 45)
    return sk.transform.rotate(image_array, random_degree)

def random_noise(image_array: ndarray, var):
    # add random noise to the image using gaussian, can also be changed so salt and pepper for example: (s&p, )
    return sk.util.random_noise(image_array, mode = 'gaussian', var = var)