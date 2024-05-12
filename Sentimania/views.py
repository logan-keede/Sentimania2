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
import os
import shutil


def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (48, 48))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    # img = np.expand_dims(img, axis=0)
    return img

def home(request):
    return render(request, "home.html")

def result(request,image_names):
    model = tf.saved_model.load('./Emotion_detector')
    images= []
    context = {'predictions': []}

    for input_image in image_names:
        input_image = preprocess_image(input_image)
        images.append(input_image)
        # model
        # predictions = model(input_image)
    images_array = np.array(images)

# Make predictions for all images at once
    predictions = model(images_array)

    context['predictions']=predictions
    dirpath = f'static/photos/'
    for filename in os.listdir(dirpath):
        filepath = os.path.join(dirpath, filename)
        try:
            shutil.rmtree(filepath)
        except OSError:
            os.remove(filepath)

    return render(request, 'result.html', context)

def camera(request):
    if request.method == 'POST':
        image_paths = request.POST.getlist("photos[]")  # Assuming src[] is the name attribute of your photo elements
        image_names = []
        for i, image_path in enumerate(image_paths):
            temp_image = NamedTemporaryFile()
            temp_image.write(urlopen(image_path).read())
            temp_image.flush()
            name = f"temp_image_{i}.jpg"  # Naming each image uniquely
            
            with open(name, 'wb') as file:
                file.write(urlopen(image_path).read())
            name = default_storage.save(f'static/photos/{name}', ContentFile(urlopen(image_path).read()))
            image_names.append(name)

        # print(l)
        return result(request,image_names)
    # return render(request, 'result.html')
