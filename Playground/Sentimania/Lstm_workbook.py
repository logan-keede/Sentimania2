import keras
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Dense, TimeDistributed
from keras.layers import LSTM, Bidirectional, Conv1D, concatenate, Permute, Dropout
from keras.preprocessing.image import ImageDataGenerator

batch_size = 32
num_classes = 10
epochs = 2

row_hidden = 128
col_hidden = 128

# row, col = X_train.shape[1:]
row = 48
col = 48
input = Input(shape=(row, col,3))
train_gen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    # rotation_range=10,  # Rotate images randomly up to 20 degrees
    width_shift_range=0.05,  # Shift width of images randomly up to 10%
    height_shift_range=0.05,  # Shift height of images randomly up to 10%
    # shear_range=0.1,  # Apply shear transformation randomly up to 10%
    zoom_range=0.05,  # Zoom images randomly up to 10%
    brightness_range=[0.9, 1.1],
    # validation_split=0.4,
    # preprocessing_function = tf.image.grayscale_to_rgb
)

train_generator = train_gen.flow_from_directory(
    directory=r"./train/",
    target_size=(48, 48),
    color_mode="rgb",
    batch_size=64,
    class_mode="categorical",
    shuffle=True,
    seed=42,
    # subset='training'
)

test_gen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    # validation_split=0.4,
    # preprocessing_function = tf.image.grayscale_to_rgb
)
valid_generator = test_gen.flow_from_directory(
    directory="test",
    target_size=(48, 48),
    color_mode="rgb",
    batch_size=64,
    class_mode="categorical",
    shuffle=False,
    seed=42,
    # subset='validation'
)
def lstm_pipe(in_layer):
    x = Conv1D(row_hidden, kernel_size=3, padding = 'same')(in_layer)
    x = Conv1D(row_hidden, kernel_size=3, padding = 'same')(x)
    # x = Conv1D(row_hidden, kernel_size=3, padding='same')(x)
    encoded_rows = Bidirectional(LSTM(row_hidden, return_sequences = True))(x)
    return LSTM(col_hidden)(encoded_rows)
# read it by rows
first_read = lstm_pipe(input)
# read it by columns
# trans_read = lstm_pipe(Permute(dims = (2,1))(input))
# encoded_columns = concatenate([first_read])
encoded_columns = Dropout(0.2)(first_read)
prediction = Dense(num_classes, activation='softmax')(encoded_columns)
model = Model(input, prediction)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()

history = model.fit(train_generator,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=valid_generator)
