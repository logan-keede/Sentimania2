import numpy
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from FaceMesh import FaceMeshDetector
import cv2
import keras.backend as K
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_class_weight
import keras.optimizers
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, Model

from keras.regularizers import l2

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
train_dir = 'train'
test_dir = 'test'
#
# #

train_gen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    # rotation_range=10,  # Rotate images randomly up to 20 degrees
    # width_shift_range=0.1,  # Shift width of images randomly up to 10%
    # height_shift_range=0.1,  # Shift height of images randomly up to 10%
    # # shear_range=0.1,  # Apply shear transformation randomly up to 10%
    # zoom_range=0.1,  # Zoom images randomly up to 10%
    # brightness_range=[0.8, 1.2],
    # validation_split=0.4,
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
# test_gen = ImageDataGenerator(
#     rescale=1./255,
#     horizontal_flip=True,
#     validation_split=0.4,
#     # preprocessing_function = tf.image.grayscale_to_rgb
# )
valid_generator = train_gen.flow_from_directory(
    directory=test_dir,
    target_size=(48, 48),
    color_mode="rgb",
    batch_size=64,
    class_mode="categorical",
    shuffle=False,
    seed=42,
    # subset='validation'
)
learning_rate = 0.0001
print(valid_generator.class_indices, train_generator.class_indices)
# x_test, y_test = detector.process_dataset(test_dir)
# np.savez("x_test.npz", matrix=x_test)
# np.savez("y_test.npz", matrix=y_test)
#
# x_train, y_train = detector.process_dataset(train_dir)
#
# np.savez("y_train.npz", matrix=y_train)
# np.savez("x_train.npz", matrix=x_train)
#
#

# #
# x_test = np.load("x_test.npz")["matrix"]
# y_test = np.load("y_test.npz")["matrix"]
# x_train = np.load("x_train.npz")["matrix"]
# y_train = np.load("y_train.npz")["matrix"]
#
# num_classes = 7
# y_train = label_encoder.fit_transform(y_train)
# y_test = label_encoder.transform(y_test)
# y_test = to_categorical(y_test)
# y_train = to_categorical(y_train)

# y_test = numpy.where(y_test==0, 0, 1)
# y_train = numpy.where(y_train==0, 0, 1)
# train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
# test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

# target_dist = [0.142,0.142,0.142,0.142,0.142,0.142,0.142]

# dataset = train_dataset.rejection_resample(
#    class_func=lambda x,y: y,
#    target_dist=target_dist,
#    # initial_dist=[0.08418,0.9158]
# )
# dataset = dataset.map(lambda class_func_result, data: data)

# dataset = dataset.map(lambda class_func_result, data: data)
# zero, one = np.bincount(list(dataset.as_numpy_iterator())) / n

# class_weights = compute_class_weight(class_weight="balanced", classes=numpy.array([0,1,2,3,4,5,6]),
#                                      y=y_train)

# di = {}
# for i in range(7):
#     di[i] = class_weights[i]

# num_features = x_train.shape[1]
# print(num_features)
# print(*y_test)
# weighted = Counter(y_train)
# for i in weighted:
#     weighted[i]= 1000/weighted[i]


# BATCH_SIZE = 64
# SHUFFLE_BUFFER_SIZE = 5000
#
# train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
# train_dataset = train_dataset.repeat()

# dataset_size = tf.data.experimental.cardinality(train_dataset).numpy()
# print("Number of elements in the dataset:", dataset_size)
#
# # Print the structure of the elements in the dataset
# print("Element spec of the dataset:", train_dataset.element_spec)
# test_dataset = test_dataset.batch(BATCH_SIZE)
# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(num_features, activation=tf.keras.layers.LeakyReLU()),
#     tf.keras.layers.Dense(512, activation=tf.keras.layers.LeakyReLU()),
#
#     tf.keras.layers.Dense(512, activation=tf.keras.layers.LeakyReLU()),
#
#     tf.keras.layers.Dense(512, activation=tf.keras.layers.LeakyReLU()),
#     tf.keras.layers.Dropout(0.4),
#     tf.keras.layers.Dense(7,activation="softmax")
# ])
# input_tensor = Input(shape=(48, 48, 1))
#
# # Convert grayscale to 3 channels
# x = Conv2D(3, (3, 3), padding='same')(input_tensor)

# Base model (EfficientNetB0)
# base_model = efn.EfficientNetB1(input_shape=(48, 48, 3), include_top=False, weights='imagenet')
#
# # Freeze the layers
# for layer in base_model.layers:
#     layer.trainable = False
# for i in range(-36,0,1):
#     base_model.layers[i].trainable = True


# Connect the input and base_model
# x = base_model.output
#
# # Global average pooling layer (you may also use Flatten())
# x = Flatten()(x)
#
# # Fully connected layers
# x = Dense(2048, activation="relu")(x)
# x = Dense(1024, activation="relu")(x)
# x = Dropout(0.5)(x)

# Output layer
def get_performance_model(leaky_relu_slope,
                          dropout_rate,
                          regularization_rate,
                          input_shape = (48, 48, 3),
                          n_classes = 7,
                          logits = False):

    regularization = l2(regularization_rate)

    model = keras.Sequential([
        layers.SeparableConv2D(48,
                               (3, 3),
                               kernel_regularizer = regularization,
                               padding = 'same',
                               input_shape = input_shape),
        layers.BatchNormalization(),
        layers.LeakyReLU(leaky_relu_slope),
        layers.SeparableConv2D(48,
                               (3, 3),
                               kernel_regularizer = regularization,
                               padding = 'same'),
        layers.BatchNormalization(),
        layers.LeakyReLU(leaky_relu_slope),
        layers.SeparableConv2D(48,
                               (3, 3),
                               kernel_regularizer = regularization,
                               padding = 'same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.SpatialDropout2D(dropout_rate),
        layers.LeakyReLU(leaky_relu_slope),

        layers.SeparableConv2D(96,
                               (3, 3),
                               kernel_regularizer = regularization,
                               padding = 'same'),
        layers.BatchNormalization(),
        layers.LeakyReLU(leaky_relu_slope),
        layers.SeparableConv2D(96,
                               (3, 3),
                               kernel_regularizer = regularization,
                               padding = 'same'),
        layers.BatchNormalization(),
        layers.LeakyReLU(leaky_relu_slope),
        layers.SeparableConv2D(96,
                               (3, 3),
                               kernel_regularizer = regularization,
                               padding = 'same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.SpatialDropout2D(dropout_rate),
        layers.LeakyReLU(leaky_relu_slope),

        layers.SeparableConv2D(192,
                               (3, 3),
                               kernel_regularizer = regularization,
                               padding = 'same'),
        layers.BatchNormalization(),
        layers.LeakyReLU(leaky_relu_slope),
        layers.SeparableConv2D(192,
                               (3, 3),
                               kernel_regularizer = regularization,
                               padding = 'same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.SpatialDropout2D(dropout_rate),
        layers.LeakyReLU(leaky_relu_slope),

        layers.SeparableConv2D(384,
                               (3, 3),
                               kernel_regularizer = regularization,
                               padding = 'same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.SpatialDropout2D(dropout_rate),
        layers.LeakyReLU(leaky_relu_slope),

        layers.SeparableConv2D(n_classes, (1, 1), padding = 'same'),
        layers.GlobalAveragePooling2D()
    ])

    if not logits:
        model.append(layers.Softmax())

    return model

BATCH_SIZE = 64
EPOCHS = 1000
DROPOUT_RATE = 0.10
LEARNING_RATE = 0.005
LEAKY_RELU_SLOPE = 0.02
LR_PATIENCE = 10
PATIENCE = 15
REGULARIZATION_RATE = 0.01

model = get_performance_model(leaky_relu_slope = LEAKY_RELU_SLOPE,
                                     dropout_rate = DROPOUT_RATE,
                                     regularization_rate = REGULARIZATION_RATE,
                                     n_classes = 7,
                                     logits = True)

loss = keras.losses.CategoricalCrossentropy(from_logits = True)

# Optimizer
model.compile(optimizer = keras.optimizers.Adam(learning_rate = LEARNING_RATE),
              loss = loss,
              metrics = ['accuracy'])
early_stopping = keras.callbacks.EarlyStopping(
    monitor = 'val_loss',
    patience = PATIENCE,
    min_delta = 0.001
)

# Learning rate reduction
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss',
                                              factor = 0.5,
                                              patience = LR_PATIENCE,
                                              verbose = 1,
                                              min_delta = 0.001)
# model_dir = os.path.join(base_dir, 'model')
# Save the model with highest validation accuracy
model_file_path_acc = os.path.join('Emotion_tracker_copy' +'_acc' + '.h5')
save_best_acc = keras.callbacks.ModelCheckpoint(model_file_path_acc,
                                                monitor = 'val_accuracy',
                                                save_best_only = True)
# Save also the model with lowest validation loss
model_file_path_loss = os.path.join('Emotion_tracker_copy' + '_loss' + '.h5')
save_best_loss = keras.callbacks.ModelCheckpoint(model_file_path_loss,
                                                monitor = 'val_loss',
                                                save_best_only = True)
# predictions = Dense(7, activation="softmax")

# Create the final model
history = model.fit(
            train_generator,
            validation_data = valid_generator,
            epochs = EPOCHS,
            callbacks = [early_stopping,
                         reduce_lr,
                         save_best_acc,
                         save_best_loss]
)
model.save("Emotion_detector_copy")
# tf.keras.backend.clear_session()
# y_train = to_categorical(y_train, num_classes=num_classes)
# y_test = to_categorical(y_test, num_classes=num_classes)
# # Train the model using fit_generator
# model.fit(x_train, y_train, epochs=40, batch_size= 32,validation_data=(x_test, y_test))