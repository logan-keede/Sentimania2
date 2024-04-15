import cv2
import numpy as np
from django.http import HttpResponse
from django.shortcuts import render
import tensorflow as tf

def preprocess_image(image_path):
    
    img = cv2.imread(image_path)
    img = cv2.resize(img, (48, 48))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def home(request):
    return render(request, "home.html")

def result(request):
    model = tf.saved_model.load('./Emotion_detector')
    input_image = preprocess_image('sample.jpg')
    predictions = model(input_image)
    context = {'predictions': predictions}
    return render(request, 'result.html', context)
