import cv2 as cv
# import tensorflow as tf
# from tensorflow.keras.preprocessing import image
import numpy as np
from FaceMesh import FaceMeshDetector
import pandas as pd

detector = FaceMeshDetector()
# model = tf.keras.models.load_model('Emotion_detector')
imgpath = "./test/sad/PrivateTest_528072.jpg"
img = cv.imread(imgpath)
cv.imshow("Display window", img)
# img = image.load_img(imgpath, target_size = (48, 48))
# cv.imshow(img)
detector.FaceMesh(img)
# img = image.img_to_array(img)
# img = np.expand_dims(img, axis = 0)

# cv2.destroyAllWindows()
# k = model.predict(img)
# print(k)