from FaceMesh import FaceMeshDetector
import numpy as np
import pandas as pd

detector = FaceMeshDetector()
test_dir = "test"
train_dir = "train"
x_test, y_test = detector.process_dataset(test_dir)
# np.savez("x_test.npz", matrix=x_test)
# np.savez("y_test.npz", matrix=y_test)
#
# x_train, y_train = detector.process_dataset(train_dir)
#
# np.savez("y_train.npz", matrix=y_train)
# np.savez("x_train.npz", matrix=x_train)


x_test = np.load("x_test.npz")["matrix"]
y_test = np.load("y_test.npz")["matrix"]
x_train = np.load("x_train.npz")["matrix"]
y_train = np.load("y_train.npz")["matrix"]
#
# create a dummy array
# arr = np.arange(1, 11).reshape(2, 5)
# print(arr)

# convert array into dataframe
DF = pd.DataFrame(x_test)


# save the dataframe as a csv file
DF.to_csv("x_test.csv")

DF = pd.DataFrame(y_test)


# save the dataframe as a csv file
DF.to_csv("y_test.csv")

DF = pd.DataFrame(x_train)


# save the dataframe as a csv file
DF.to_csv("x_train.csv")

DF = pd.DataFrame(y_train)


# save the dataframe as a csv file
DF.to_csv("y_train.csv")
