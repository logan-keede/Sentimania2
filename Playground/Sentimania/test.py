# import sounddevice as sd
# import numpy as np
# import threading
# import time
# import speech_recognition as sr
# from scipy.io.wavfile import write
# import queue
# recognizer = sr.Recognizer()
#
#
# with sr.Microphone() as source:
#     print("Listening...")
#     audio = recognizer.listen(source)
#     try:
#         text = recognizer.recognize_google(audio)
#         print("You said:", text)
#     except sr.UnknownValueError:
#         print("Sorry, I could not understand.")


import cv2

from fer import FER

input_image = cv2.imread("smile.jpg")
emotion_detector = FER()
# Output image's information
print(emotion_detector.detect_emotions(input_image))