import cv2
import numpy as np
from django.http import HttpResponse
from django.shortcuts import render
import tensorflow as tf
# import sys
from django.shortcuts import render
from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage
# from corona.analyzer import main
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from django.core.files import File
from django.core.files.base import ContentFile
from django.core.files.temp import NamedTemporaryFile
from urllib.request import urlopen
from django.core.files.storage import default_storage

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (48, 48))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def home(request):
    return render(request, "home.html")

def result(request,l):
    model = tf.saved_model.load('./Emotion_detector')
    input_image = preprocess_image(l)
    predictions = model(input_image)
    context = {'predictions': predictions}
    return render(request, 'result.html', context)

def camera(request):
    if request.method == 'POST':
        image_path = request.POST["src"]
        image = NamedTemporaryFile()
        urlopen(image_path).read()
        image.write(urlopen(image_path).read())
        image.flush()
        image = File(image)
        name = str(image.name).split('\\')[-1]
        name += '.jpg'  # store image in jpeg format
        image.name = name
        with open('image.txt', 'w+') as file:
            file.write(str(name))
        l = default_storage.save('static/photos/temp.jpg', ContentFile(urlopen(image_path).read()))
        print(l)
        return result(request,l)
    # return render(request, 'result.html')
