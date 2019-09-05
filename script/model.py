#!/usr/bin/env python2.7
# Udacity
# Selfdriving cars Nanodegree program
# Cajazeiras - PB / Brazil
# Author: Raphaell Maciel de Sousa
# Description: program to generate a model for behavioral cloning
# Data: 07/07/2017
# Modification: 31/08/2019

# Load required packages ###########################################################################

import os
import csv
import cv2
import numpy as np
import sklearn

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Lambda, Cropping2D
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

from keras.models import load_model

# Load data for training ############################################################################

samples = []
#PATH_log = '/home/raphaell/catkin_ws_ROSI/src/rosi_defy/script/left_side/robotCommands/driving_log.csv'
#PATH_IMG = '/home/raphaell/catkin_ws_ROSI/src/rosi_defy/script/left_side/rgb_data/'
PATH_log = '/home/raphaell/catkin_ws_ROSI/src/rosi_defy/script/t4/robotCommands/driving_log.csv'
PATH_IMG = '/home/raphaell/catkin_ws_ROSI/src/rosi_defy/script/t4/rgb_data/'

lines = []
with open(PATH_log) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
#print(lines[0][2])

# Preprocessing steps: ###############################################################################
'''
For this project it will be used 3 steps:
1. First it was used a GaussianBlur to minimize noize
2. To minimize memory usage, it was used a resize function for images, the new dimension is (64,64,3)
3. After all it is used a YUV color transformation
'''
def preprocess(img):
    image = cv2.GaussianBlur(img, (3,3), 0)
    image = cv2.resize(image, (64,64), interpolation=cv2.INTER_AREA)
    proc_img = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)	# cv2 loads images as BGR
    return proc_img

#Here it is used a generator function for a low memory consum, as is explaned in Udacity class ######

def load_img():
    images = []
    joins = []
    count = 0
    for line in lines:
        #for i in range(3):
        source_path = line[0]
        filename = source_path.split('/')[-1]
        current_path = PATH_IMG + filename
        image = cv2.imread(current_path)
        #image = image[40:160,10:310,:]
        image = preprocess(image)        
        images.append(image)
        join0 = lines[count][2]
        join1 = lines[count][3]
        join2 = lines[count][4]
        join3 = lines[count][5]
        join = [join0, join1, join2, join3]
        count = count + 1
                
        joins.append(join)
    return images, joins

X_train, y_train = load_img()

X_train = np.array(X_train)
y_train = np.array(y_train)

# Nvidea model
model = Sequential()
model.add(Lambda(lambda x: x/255.0-0.5, input_shape=(64,64,3)))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(4))

#model.summary()

# Training model using adam optimizer
model.compile(loss='mse', optimizer='adam')

#model.compile(optimizer='adam', loss='mse')
model.fit(np.array(X_train), y_train, validation_split = 0.2, shuffle = True, epochs = 20)

model.save('model.h5')

# Load data for training ############################################################################

# Load trained model for a new data set
#model = load_model('model.h5')

#samples = []
#PATH_log = '/home/raphaell/catkin_ws_ROSI/src/rosi_defy/script/corredor/robotCommands/driving_log.csv'
#PATH_IMG = '/home/raphaell/catkin_ws_ROSI/src/rosi_defy/script/corredor/rgb_data/'

#lines = []
#with open(PATH_log) as csvfile:
#    reader = csv.reader(csvfile)
#    for line in reader:
#        lines.append(line)

#X_train, y_train = load_img()

#X_train = np.array(X_train)
#y_train = np.array(y_train)

## Training model using adam optimizer
#model.compile(loss='mse', optimizer='adam')

#model.compile(optimizer='adam', loss='mse')
#model.fit(np.array(X_train), y_train, validation_split = 0.2, shuffle = True, epochs = 3)

#model.save('model.h5')

#############################################################################################################################################

# Load trained model for a new data set
model = load_model('model.h5')

samples = []
PATH_log = '/home/raphaell/catkin_ws_ROSI/src/rosi_defy/script/t3/robotCommands/driving_log.csv'
PATH_IMG = '/home/raphaell/catkin_ws_ROSI/src/rosi_defy/script/t3/rgb_data/'

lines = []
with open(PATH_log) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

X_train, y_train = load_img()

X_train = np.array(X_train)
y_train = np.array(y_train)

## Training model using adam optimizer
model.compile(loss='mse', optimizer='adam')

model.compile(optimizer='adam', loss='mse')
model.fit(np.array(X_train), y_train, validation_split = 0.2, shuffle = True, epochs = 3)

model.save('model.h5')

#############################################################################################################################################

# Load trained model for a new data set
model = load_model('model.h5')

samples = []
PATH_log = '/home/raphaell/catkin_ws_ROSI/src/rosi_defy/script/t2/robotCommands/driving_log.csv'
PATH_IMG = '/home/raphaell/catkin_ws_ROSI/src/rosi_defy/script/t2/rgb_data/'

lines = []
with open(PATH_log) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

X_train, y_train = load_img()

X_train = np.array(X_train)
y_train = np.array(y_train)

## Training model using adam optimizer
model.compile(loss='mse', optimizer='adam')

model.compile(optimizer='adam', loss='mse')
model.fit(np.array(X_train), y_train, validation_split = 0.2, shuffle = True, epochs = 3)

model.save('model.h5')

#############################################################################################################################################


