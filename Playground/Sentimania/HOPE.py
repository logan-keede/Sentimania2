import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import efficientnet.keras as efn
import keras.backend as K
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, Flatten, Dropout,BatchNormalization ,Activation
from tensorflow.keras.models import Model, Sequential
from keras.applications.nasnet import NASNetLarge
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam

train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   validation_split=0.2,

                                   rotation_range=5,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   # zoom_range=0.2,
                                   horizontal_flip=True,
                                   vertical_flip=True,
                                   fill_mode='nearest')

valid_datagen = ImageDataGenerator(rescale=1. / 255,
                                   validation_split=0.2)

test_datagen = ImageDataGenerator(rescale=1. / 255
                                  )
train_dataset  = train_datagen.flow_from_directory(directory = 'train',
                                                   target_size = (48,48),
                                                   class_mode = 'categorical',
                                                   subset = 'training',
                                                   batch_size = 64)

valid_dataset = valid_datagen.flow_from_directory(directory = 'train',
                                                  target_size = (48,48),
                                                  class_mode = 'categorical',
                                                  subset = 'validation',
                                                  batch_size = 64)

test_dataset = test_datagen.flow_from_directory(directory = 'test',
                                                  target_size = (48,48),
                                                  class_mode = 'categorical',
                                                  batch_size = 64)

# from keras.preprocessing import image
# img = image.load_img("../input/fer2013/test/angry/PrivateTest_10131363.jpg",target_size=(48,48))
# img = np.array(img)
# plt.imshow(img)
# print(img.shape)
#
# img = np.expand_dims(img, axis=0)
# from keras.models import load_model
# print(img.shape)

base_model = efn.EfficientNetB0(input_shape=(48,48,3),include_top=False,weights="imagenet")
for layer in base_model.layers[:-4]:
    layer.trainable=False



model=Sequential()
model.add(base_model)
model.add(Dropout(0.5))
model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(32,kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(32,kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(32,kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(7,activation='softmax'))
# Model Summary

model.summary()

# model_final = Model(inputs=base_model.input, outputs=predictions)
lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3)
early_stopper = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0, patience=6, mode='auto')



model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, decay=1e-8), loss='categorical_crossentropy',metrics=["accuracy"])
history=model.fit(train_dataset,validation_data=valid_dataset,epochs = 50,verbose = 1, callbacks=[lr_reducer, early_stopper])
