#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

###############################################################################################################################
#			
#	This code has been developed for the ROSI Challenge 2019 https://github.com/filRocha/rosiChallenge-sbai2019
#	Team: ForROS		
#	Institutions: Federal Institute of Paraiba (Cajazeiras) and Federal Institute of Bahia	
#	Team: Raphaell Maciel de Sousa (team leader/IFPB)
#		Gerberson Felix da Silva (IFPB)	
#		Jean Carlos Palácio Santos (IFBA)
#		Rafael Silva Nogueira Pacheco (IFBA)
#		Michael Botelho Santana (IFBA)
#		Sérgio Ricardo Ferreira Andrade Júnior (IFBA)
#		Matheus Vilela Novaes (IFBA)		
#		Lucas dos Santos Ribeiro (IFBA)
#		Félix Santana Brito (IFBA)
#		José Alberto Diaz Amado (IFBA)
#
#	Approach: it was used a behavioral clonning technique to move the robot around the path and avoid obstacles.
#
###############################################################################################################################

# Load required packages
import os
import csv
import cv2
import numpy as np

from keras.models import Sequential
from keras.layers import Lambda, Cropping2D
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

from keras.models import load_model

from keras.callbacks import EarlyStopping, ModelCheckpoint

# Load data for training ############################################################################

#model = load_model('model.h5')

samples = []
PATH_log = '/home/raphaell/catkin_ws/src/rosi_defy_forros/script/robotCommands/driving_log.csv'
PATH_IMG = '/home/raphaell/catkin_ws/src/rosi_defy_forros/script/rgb_data/'

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
        source_path = line[1]
        filename = source_path.split('/')[-1]
        current_path = PATH_IMG + filename
        join0 = round(float(lines[count][2]), 5)
        join1 = round(float(lines[count][3]), 5)
        join2 = round(float(lines[count][4]), 5)
        join3 = round(float(lines[count][5]), 5)
        image = cv2.imread(current_path)
        #image = image[40:160,10:310,:]
        image = preprocess(image)        
        images.append(image)
       	join = [join0, join1, join2, join3]
	joins.append(join)
	
	count = count + 1              
        
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
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Dense(4))

model.summary()

# Set callback functions to early stop training and save the best model so far
callbacks = [EarlyStopping(monitor='accuracy', patience=2),
             ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]

# Training model using adam optimizer
# Compile model
model.compile(optimizer='adam', loss='mse')
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.fit(np.array(X_train), y_train, validation_split = 0.0, shuffle = True, epochs = 3, callbacks=callbacks)

model.save('startModel.h5')
