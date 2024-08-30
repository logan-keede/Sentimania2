from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical
from keras.layers import LSTM
from keras.utils import np_utils
import pandas as pd
import numpy as np


def precision(y_true, y_pred, normalize=True, sample_weight=None):
    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set( np.where(y_true[i])[0] )
        set_pred = set( np.where(y_pred[i])[0] )
        #print('\nset_true: {0}'.format(set_true), ', set_pred: {0}'.format(set_pred))
        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        elif len(set_pred) == 0:
            tmp_a = 0
        else:
            tmp_a = len(set_true.intersection(set_pred))/\
                    float(len(set_pred))
        acc_list.append(tmp_a)
    return np.mean(acc_list)



def recall(y_true, y_pred, normalize=True, sample_weight=None):
    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set( np.where(y_true[i])[0] )
        set_pred = set( np.where(y_pred[i])[0] )
        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred))/\
                    float(len(set_true))
        acc_list.append(tmp_a)
    return np.mean(acc_list)

def f_score(y_true, y_pred, normalize=True, sample_weight=None):
    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set( np.where(y_true[i])[0] )
        set_pred = set( np.where(y_pred[i])[0] )
        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = (2*len(set_true.intersection(set_pred)))/\
                    float( len(set_true) + len(set_pred))
        acc_list.append(tmp_a)
    return np.mean(acc_list)








label_encoder = LabelEncoder()
x_test = np.load("x_test.npz")["matrix"]
y_test = np.load("y_test.npz")["matrix"]
x_train = np.load("x_train.npz")["matrix"]
y_train = np.load("y_train.npz")["matrix"]
print(x_test.shape)
num_classes = 7

# exit()

y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)
y_test = to_categorical(y_test)
y_train = to_categorical(y_train)


def lstm_model(o_dim):
    lstm_model = Sequential()
    lstm_model.add(LSTM(128, input_shape=(1404, 1)))
    # lstm_model.add(LSTM(128))
    lstm_model.add(Dense(512))
    lstm_model.add(Dense(o_dim, activation='softmax'))
    lstm_model.compile(loss='categorical_crossentropy',
               optimizer='adam',
               metrics=['accuracy'])
    return lstm_model




model1 = lstm_model(7)
print(model1.summary())
early_stopping_monitor = EarlyStopping(monitor='val_loss',
                                       patience=5,
                                       verbose=1)
history = model1.fit(x_train,
          y_train,
          validation_data=(x_test, y_test),
          epochs = 100,
          batch_size =64,
          verbose=1,
          callbacks=[early_stopping_monitor])



y_test = y_test.ravel()
Y_test  = label_encoder.transform(y_test)
Y_test  = np_utils.to_categorical(y_test)
#Y_test=np.array(Y_test)


pred = model1.predict(x_test)
##pred=pred.todense()
#a=accuracy_score(Y_test,pred)
#print(a)
#

th = [0.1, 0.15, 0.2, 0.25, 0.3, 0.32, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
r, p, f = [],[],[]
for t in th:
    pred = model1.predict(x_test)
    pred[pred>= t] = int(1)
    pred[pred< t] = int(0)
    r.append(recall(Y_test, pred)*100)
    p.append(precision(Y_test, pred)*100)
    f.append(f_score(Y_test, pred)*100)

    #Summarize history for loss

import matplotlib.pyplot as plt
plt.plot(p)
plt.plot(f)
plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11], th)
plt.grid()
plt.ylabel('percentage')
plt.xlabel('threshold')
plt.legend(['precision','f1-score'], loc = 'upper right')
plt.show()
index = np.argmax(np.array(f))
print('Recall: ',r[index])
print('Precision: ',p[index])
print('F1-score: ',f[index] )
# """