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

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
# Define your image size and batch size
label_encoder = LabelEncoder()
detector = FaceMeshDetector()
# Define the path to your train and test directories
train_dir = 'train'
test_dir = 'test'

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
x_test = np.load("x_test.npz")["matrix"]
y_test = np.load("y_test.npz")["matrix"]
x_train = np.load("x_train.npz")["matrix"]
y_train = np.load("y_train.npz")["matrix"]

num_classes = 7
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)
y_test = to_categorical(y_test)
y_train = to_categorical(y_train)


train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

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

num_features = x_train.shape[1]
# print(num_features)
# print(*y_test)
# weighted = Counter(y_train)
# for i in weighted:
#     weighted[i]= 1000/weighted[i]


BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 5000

train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
train_dataset = train_dataset.repeat()

dataset_size = tf.data.experimental.cardinality(train_dataset).numpy()
print("Number of elements in the dataset:", dataset_size)

# Print the structure of the elements in the dataset
print("Element spec of the dataset:", train_dataset.element_spec)
test_dataset = test_dataset.batch(BATCH_SIZE)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(num_features, activation=tf.keras.layers.LeakyReLU()),
    tf.keras.layers.Dense(512, activation=tf.keras.layers.LeakyReLU()),

    tf.keras.layers.Dense(512, activation=tf.keras.layers.LeakyReLU()),

    tf.keras.layers.Dense(512, activation=tf.keras.layers.LeakyReLU()),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(7,activation="softmax")
])
input_tensor = Input(shape=(48, 48, 1))
model.compile(optimizer=optimizers.Adam(learning_rate=0.0001, decay=1e-8),
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

# y_train = to_categorical(y_train, num_classes=num_classes)
# y_test = to_categorical(y_test, num_classes=num_classes)
# # Train the model using fit_generator
# model.fit(x_train, y_train, epochs=40, batch_size= 32,validation_data=(x_test, y_test))
# # Convert grayscale to 3 channels
# x = Conv2D(3, (3, 3), padding='same')(input_tensor)
