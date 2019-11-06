import numpy as np
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense, Flatten
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt

train_path = '/mnt/Data/leite/test-keras-vgg16/train'
valid_path = '/mnt/Data/leite/test-keras-vgg16/valid'
test_path =  '/mnt/Data/leite/test-keras-vgg16/test'

train_batches = ImageDataGenerator().flow_from_directory(train_path,
                                                         target_size=(224, 224),
                                                         classes=['Falls', 'NotFalls'],
                                                         batch_size=10)

valid_batches = ImageDataGenerator().flow_from_directory(valid_path,
                                                         target_size=(224, 224),
                                                         classes=['Falls', 'NotFalls'],
                                                         batch_size=4)

test_batches = ImageDataGenerator().flow_from_directory(test_path,
                                                         target_size=(224, 224),
                                                         classes=['Falls', 'NotFalls'],
                                                         batch_size=10)

''' VGG16 '''
vgg16_model = keras.applications.vgg16.VGG16()
vgg16_model.summary()

type(vgg16_model)

model = Sequential()
for layer in vgg16_model.layers:
    model.add(layer)

model.summary()

model.layers.pop()

model.summary()

for layer in model.layers:
    layer.trainable = False

model.add(Dense(2, activation='softmax'))
model.summary()

model.compile(Adam(lr=.00001), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit_generator(train_batches, steps_per_epoch=4, validation_data=valid_batches, validation_steps=4, epochs=10, verbose=2)

test_imgs, test_labels = next(test_batches)
test_labels = test_labels[:, 0]

predictions = model.predict_generator(test_batches, steps=1, verbose=0)

print(test_labels)
print(np.round(predictions[:,0]))
cm = confusion_matrix(test_labels, np.round(predictions[:, 0]))
print(cm)
