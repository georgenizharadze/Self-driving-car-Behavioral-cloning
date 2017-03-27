# Import required modules
import csv
import cv2
import numpy as np
import keras
from keras.models import Sequential, Model
from keras.layers import Cropping2D
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
import sklearn 
from sklearn.utils import shuffle 
from sklearn.model_selection import train_test_split
import os
from time import time
import random

t = time()

#read in 'smooth driving' data, i.e. driving normally in the middle of the road
lines = []
with open('./my_data_smooth_driving/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)       

images = []
measurements = []

for line in lines[:]:
    original_path = line[0]
    components = original_path.split('\\')
    filename = components[-1]
    new_path = 'my_data_smooth_driving/IMG/' + filename  
    image = cv2.imread(new_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)

print (len(images), 'images')
print (len(measurements), 'steerting angle values')

# read in data from recovery driving, i.e. driving away from road edges, etc.
lines_recovery = []
with open('./my_data_recovery_situations_6/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines_recovery.append(line)  

images_recovery = []
measurements_recovery = []
        
for line in lines_recovery:
    original_path = line[0]
    components = original_path.split('\\')
    filename = components[-1]
    new_path = 'my_data_recovery_situations_6/IMG/' + filename  
    image = cv2.imread(new_path)
    images_recovery.append(image)
    measurement = float(line[3]) 
    measurements_recovery.append(measurement)

print (len(images_recovery), 'recovery situation images')
print (len(measurements_recovery), 'recovery situation measurements')    

# combine two data sets and create arrays for model training
X_train = np.array(images +images_recovery)
y_train = np.array(measurements + measurements_recovery)

print (len(X_train), 'combined X length')
print (len(y_train), 'combined y length')  

print(time() - t, 'seconds')

# implement and train a model
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape = (160, 320, 3)))
model.add(Cropping2D(cropping=((70,20), (0,0))))
model.add(Convolution2D(6,(5,5), activation='relu'))
model.add(Dropout(0.25)) 
model.add(Convolution2D(16,(5,5), activation='relu'))
model.add(Dropout(0.25)) 
model.add(Flatten())
model.add(Dense(120))
model.add(Activation('relu'))
model.add(Dense(84))
model.add(Activation('relu'))
model.add(Dense(1))


model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=3)
model.save('model_20.h5')
print('Model saved.') #save the model
