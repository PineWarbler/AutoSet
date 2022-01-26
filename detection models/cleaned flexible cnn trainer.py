# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 19:37:07 2021

@author: Peter Reynolds
"""

import numpy as np
import time
import pandas as pd
from matplotlib.image import imread

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


tf.autograph.set_verbosity(1) # decrease logging level

AddressOnlySource = '<folder of training images>/' # replace with actual file path
FileTypeOnlySource = '.jpg'

allImagesinArray = [] # create empty list to compile training images

# define standardized dimensions for training images (180 x 135 x 3)
xDimensionsofPhotos = 180
yDimensionsofPhotos = 135
numColorChannels = 3 # 3 for RGB photos, 1 for grayscale photos
inputDimensions = (xDimensionsofPhotos, yDimensionsofPhotos, numColorChannels)

numClasses = 3 # number of possible classes [e.g. three classes because shape could be oval, diamond, or squiggle]

# load training spreadsheet that contains labels for the training images
data = pd.read_csv('<path to training spreadsheet>.csv', skiprows=0)

def formatImage(img, mode='float32'): 
    '''This function formats an image to conform to a given data format'''
    if mode=='float32':
        # converts image to float32 from 0-1 range
        if np.amax(img) > 1:
            img = img/255 # convert to interval [0, 1]
        if type(img) != 'float32':
            img = img.astype('float32')
        return img
    elif mode=='uint8':
        # converts image to uint8 from 0-255 range
        if np.mean(img) < 1:
            img = img*255 # convert to interval [0,255]
        if type(img) != 'uint8':
            img = img.astype('uint8')
        return img
    else:
        # inputted mode is not a valid option, so return the original image
        return img

#load images based on order of training spreadsheet entry, not order in folder (to avoid alphabetization errors)
for i in range(0, len(data.index)):
    CurrentImage = imread(AddressOnlySource + (data['filename'][i])) # read in the next image

    # make sure that image is in the correct data format
    CurrentImage = formatImage(CurrentImage, mode="float32")

    allImagesinArray.append(np.asarray(CurrentImage)) # append current image to master list of images

    del CurrentImage # delete content of image variable to save memory when loading thousands of images
    

allImagesinArray = np.asarray(allImagesinArray) # convert list to numpy array

print('Done compiling training photos.')
print('allImagesinArray shape:', allImagesinArray.shape)


# using `color` attribute as an example (but could alternatively be `Shape`, `shade`, `number`)
labels = np.asarray(data.color) # get string descriptors from training spreadsheet

fittedLabels = LabelEncoder().fit_transform(labels) # integer encode classes
print(fittedLabels) # coded labels as integers

categorizedLabels = to_categorical(fittedLabels, num_classes=numClasses) # convert integer-encoded labels to one-hot encoded labels
print('one-hot encoded labels:', categorizedLabels)


# divide images into train and test groups (70/30 split)
picture_train, picture_test, labels_train, labels_test = train_test_split(allImagesinArray, categorizedLabels, test_size=0.3, shuffle=True)


# define model
model = keras.Sequential(
    [
        keras.Input(shape=inputDimensions),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dense(14),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(numClasses, activation="softmax"),
    ]
)

# define an on-the-go data generator to augment the training set with varied images
datagen = ImageDataGenerator(
    featurewise_center=True,
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    data_format='channels_last',
    zoom_range=(0.9, 1.1),
    brightness_range=(0.8, 1.2),
    fill_mode='nearest')

datagen.fit(picture_train)

model.summary() # print model information
batch_size = 50 # number of images to train with at once
epochs = 100 # number of training epochs

start = time.time() # begin clock to time model training

# compile the model
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["categorical_accuracy"])

# train the model
model.fit(datagen.flow(picture_train, labels_train, batch_size=10), batch_size=batch_size, epochs=epochs)

# test the model
score = model.evaluate(picture_test, labels_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

end = time.time()

# OPTIONAL: only save model if its accuracy exceeds a threshold
scoreThreshold = 0.97
if score[1] >= scoreThreshold:
    model.save('<model name>.h5')
    print("Saved model to disk.")

print('training and testing runtime is:', end-start, "seconds") # print runtime