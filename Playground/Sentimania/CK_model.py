import numpy
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from FaceMesh import FaceMeshDetector
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_class_weight
import keras.optimizers
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Flatten, Dense, Dropout, Input,Conv2D
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
import efficientnet.keras as efn
# import keras.callbacks.
from collections import Counter

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
# Define your image size and batch size
label_encoder = LabelEncoder()
detector = FaceMeshDetector()
# Define the path to your train and test directories
train_dir = 'CK'
# test_dir = 'test'
#
# #

train_gen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    # rotation_range=10,  # Rotate images randomly up to 20 degrees
    width_shift_range=0.1,  # Shift width of images randomly up to 10%
    height_shift_range=0.1,  # Shift height of images randomly up to 10%
    # shear_range=0.1,  # Apply shear transformation randomly up to 10%
    zoom_range=0.1,  # Zoom images randomly up to 10%
    brightness_range=[0.9, 1.1],
    validation_split=0.2,
    # preprocessing_function = tf.image.grayscale_to_rgb
)

train_generator = train_gen.flow_from_directory(
    directory=train_dir,
    target_size=(48, 48),
    color_mode="rgb",
    batch_size=64,
    class_mode="categorical",
    shuffle=True,
    seed=42,
    subset='training'
)
test_generator = train_gen.flow_from_directory(
    directory=train_dir,
    target_size=(48, 48),
    color_mode="rgb",
    batch_size=64,
    class_mode="categorical",
    shuffle=True,
    seed=42,
    subset='validation'
)

learning_rate = 0.0001

# Base model (EfficientNetB0)
base_model = efn.EfficientNetB1(input_shape=(48, 48, 3), include_top=False, weights='imagenet')

# Freeze the layers
for layer in base_model.layers:
    layer.trainable = False
for i in range(-28,0,1):
    base_model.layers[i].trainable = True


# Connect the input and base_model
x = base_model.output

# Global average pooling layer (you may also use Flatten())
x = Flatten()(x)

# Fully connected layers
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)

# Output layer
predictions = Dense(7, activation="softmax")(x)

# Create the final model
model_final = Model(inputs=base_model.input, outputs=predictions)
lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3)
early_stopper = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0, patience=6, mode='auto')

# weight = list(map(lambda x: weighted[x], y_train))
# weight = Counter(y_train)
# y_train[0], y_train[1] = y_train[1]/y_train[1], y_train[0]/y_train[1]
# cce = tf.losses.BinaryCrossentropy(from_logits=False)
#
# def custom_loss(y_true, y_pred):
#     weight_fn = lambda x: 10.879 if x == 0.0 else 1.0
#     y_true_float = tf.cast(y_true, tf.float32)
#     y_pred = tf.cast(y_pred, tf.float32)
#     binary_loss = tf.keras.losses.binary_crossentropy(y_true_float, y_pred)
#     weighted_loss = tf.reduce_mean(tf.multiply(binary_loss, tf.map_fn(weight_fn, y_true_float)))
#     return weighted_loss

model_final.compile(optimizer=optimizers.Adam(learning_rate=learning_rate, decay=1e-8),
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
# model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.00001), loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
# model.summary()
model_final.fit(train_generator , epochs=100, validation_data=test_generator, callbacks=[lr_reducer, early_stopper])
# print(model.predict(train_dataset))
# model.evaluate(test_dataset)
model_final.save("Emotion_detector CK")
tf.keras.backend.clear_session()
# y_train = to_categorical(y_train, num_classes=num_classes)
# y_test = to_categorical(y_test, num_classes=num_classes)
# # Train the model using fit_generator
# model.fit(x_train, y_train, epochs=40, batch_size= 32,validation_data=(x_test, y_test))