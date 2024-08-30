import cv2 as cv
import dlib
# import time
import mediapipe as mp
import numpy as np
import sounddevice as sd
import numpy as np
import threading
import time
import speech_recognition as sr
from scipy.io.wavfile import write
import queue
import os
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
# plt.style.use('seaborn-poster')
import time
import mediapipe.python.solutions.face_mesh_connections as connections
left_eye = [i for i, j in connections.FACEMESH_LEFT_EYE]
right_eye = [i for i, j in connections.FACEMESH_RIGHT_EYE]
left_eye_brow = [i for i, j in connections.FACEMESH_LEFT_EYEBROW]
right_eye_brow = [i for i, j in connections.FACEMESH_RIGHT_EYEBROW]
lips = [i for i, j in connections.FACEMESH_LIPS]
nose = [i for i, j in connections.FACEMESH_NOSE]
landmark = left_eye+right_eye+left_eye_brow+right_eye_brow+lips+nose






# from fer import FER

def record_audio(output_queue):
    # frames = int(fs * duration)
    # recording = sd.rec(frames, samplerate=fs, channels=2, dtype=np.int32)
    # sd.wait()
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)
    output_queue.put(audio)


def recognize_speech_wav(audio):
    recognizer = sr.Recognizer()

    # with sr.AudioFile(file_path) as source:
    #     recognizer.adjust_for_ambient_noise(source)
    #     print("Recognizing speech...")
    #     audio_data = recognizer.record(source)
    #
    #     try:
    #         text = recognizer.recognize_google(audio_data)
    #         print("Speech recognized:", text)
    #
    #     except sr.UnknownValueError:
    #         print("Speech Recognition could not understand audio")
    #
    #     except sr.RequestError as e:
    #         print(f"Could not request results from Google Web Speech API; {e}")
    try:
        text = recognizer.recognize_google(audio)
        print("You said:", text)
    except sr.UnknownValueError:
        print("Sorry, I could not understand.")


mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=1)

drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)


# while True:
class FaceMeshDetector:
    def __init__(
        self,
        static_image_mode=False,
        max_num_faces=4,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ):
        self.static_image_mode = static_image_mode
        self.max_num_faces = max_num_faces
        self.refine_landmarks = refine_landmarks
        self.min_tracking_confidence = min_tracking_confidence
        self.min_detection_confidence = min_detection_confidence
        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = mpFaceMesh.FaceMesh(max_num_faces=1)

        self.drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)
        # self.face_detector = dlib.get_frontal_face_detector()
        #
        # self.face_cascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
        # self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    def shape_to_np(self, shape, dtype="int"):
        coords = np.zeros((68, 2), dtype=dtype)
        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
        return coords

    def FaceMesh(self, img, land = 136589):
        #
        try:
            imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        except:
            return []
        results = self.faceMesh.process(imgRGB)
        matrix = []
        ih, iw, ic = img.shape
        img_len = (ih * ih + iw * iw) ** 0.5
        if results.multi_face_landmarks:
            for face in results.multi_face_landmarks:
                self.mpDraw.draw_landmarks(
                    img, face, mpFaceMesh.FACEMESH_FACE_OVAL, drawSpec, drawSpec
                )
                for id, lm in enumerate(face.landmark):
                    x, y, z = lm.x, lm.y, lm.z
                    # print(x,y,z)
                    matrix.append((x, y, z))

        # return matrix
        if matrix ==[]:
            return matrix
            # return list(map(lambda x: x, matrix))
        # print(list(map(lambda x: x[0], matrix)))
        X = np.array(list(map(lambda x: x[0], matrix)))
        Y = np.array(list(map(lambda x: x[1], matrix)))
        Z = np.array(list(map(lambda x: x[2], matrix)))
        fig = plt.figure(figsize=(10, 10))
        ax = plt.axes(projection='3d')
        ax.grid()

        ax.scatter(X, Y, Z, c='r', s=4)
        ax.set_title('3D Scatter Plot')

        plt.show()

        # plt.show()
        time.sleep(0.5)


        # std = matrix[1]
        # dp=[]
        # for i in list(map(lambda x: (x[0]-std[0],x[1]-std[1],x[2]-std[2]), matrix)):
        #     dp.extend(i)
        # return dp
        # matrix1 = list(map(lambda x: x[1], matrix))
        # matrix2 =
        # dp = [0]*(land)

        dp = []
        for i in landmark:
            for j in landmark:
                dp.append((abs(matrix[j][1]-matrix[i][1])**2+abs(matrix[j][2]-matrix[i][2])**2)**0.5)
        # for i in matrix:
        #     dp.extend(i[1::])
        return dp

        # faces = face_cascade.detectMultiScale(gray, scaleFactor=1.10, minNeighbors=5)
        # looping through each detected faces and drawing rectangle around the face and
        # try:
        #     faces = self.face_detector(gray,1)
        #     shape1 = self.predictor(gray, faces[0])
        #     return self.shape_to_np(shape1)
        # except:
        #     return []
        # shape1 = predictor(gray, faces[0])
        # circles around the feature points
        # if len(faces) > 0:
        #     for x, y, w, h in faces:
        #         # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)
        #         # creating the rectangle object from the outputs of haar cascade calssifier
        #
        #         drect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
        #         landmarks = predictor(gray, drect)
        #         points = self.shape_to_np(landmarks)
        #         return points

    def process_directory(self, directory_path):
        # Iterate through all images in the directory
        # Iterate through all images in the directory
        data = []
        labels = []
        count = 0
        for filename in os.listdir(directory_path):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                img_path = os.path.join(directory_path, filename)
                img = cv.imread(img_path)
                landmarks = self.FaceMesh(img)
                # print(len(landmarks))
                if landmarks!=[]:  # Check if any landmarks were detected
                    data.append(landmarks)
                    labels.append(os.path.basename(directory_path))
                else:
                    count += 1
        print("count,", count)
        print(os.path.basename(directory_path))
        return data, labels

    def process_dataset(self, dataset_path):
        data = []
        labels = []
        # print()
        for label in os.listdir(dataset_path):
            if os.path.isdir(os.path.join(dataset_path, label)):
                label_path = os.path.join(dataset_path, label)
                img_data, img_labels = self.process_directory(label_path)
                data.extend(img_data)
                labels.extend(img_labels)
        return np.array(data, dtype=np.float32), np.array(labels)


def main():
    path = "C:\\Users\\HP\\PycharmProjects\\Sentimania\\VEATIC\\videos\\10.mp4"
    pTime = 0
    cap = cv.VideoCapture(0)
    detector = FaceMeshDetector()
    matrix = []
    seconds = 5  # Duration of recording
    fs = 44100  # Sample rate
    output_queue = queue.Queue()
    record_audio(output_queue)
    recording = output_queue.get()
    # emotion_detector = FER()

    try:
        # Start recording thread in the background
        # record_thread = threading.Thread(target=record_audio, args=(fs, seconds, output_queue))
        # record_thread.start()
        # wav_file_path = 'output.wav'
        while True:
            seconds = 5  # Duration of recording
            fs = 44100  # Sample rate
            # output_queue = queue.Queue()

            # Start recording thread in the background
            record_thread = threading.Thread(
                target=record_audio, args=[output_queue]
            )

            recognize_thread = threading.Thread(
                target=recognize_speech_wav, args=[recording]
            )
            record_thread.start()
            recognize_thread.start()
            print("Recording in the background...")
            Time = time.time()
            # recognize_thread = threading.Thread(target=recognize_speech_wav, args=(wav_file_path))
            while record_thread.is_alive():
                suc, img = cap.read()
                cTime = time.time()
                fps = 1 / (cTime - pTime)
                pTime = cTime

                cv.putText(
                    img,
                    f"{int(fps)}",
                    (20, 70),
                    cv.FONT_HERSHEY_PLAIN,
                    3,
                    (0, 255, 0),
                    3,
                )
                mat = detector.FaceMesh(img)
                if len(mat) == 468:
                    matrix.append(mat)
                cv.imshow("Image", img)
                # print(emotion_detector.detect_emotions(img))
                key = cv.waitKeyEx(1)
                if key == ord("q") or key == 27:  # 'q' or Esc key
                    break
            recognize_thread.join()
            record_thread.join()

            # Get the recorded data from the queue
            recording = output_queue.get()

            # Save recorded audio as a PCM WAV file
            # write('output.wav', fs, recording)

            # Perform speech recognition on the saved WAV file
            # wav_file_path = 'output.wav'
            # recognize_speech_wav(wav_file_path)
            key = cv.waitKeyEx(1)
            if key == ord("q") or key == 27:  # 'q' or Esc key
                break

    except KeyboardInterrupt:
        print("KeyboardInterrupt: Exiting the program.")
    finally:
        # Release the video capture and close the OpenCV window
        cap.release()
        cv.destroyAllWindows()

        np.savez("game_data.npz", matrix=matrix)


main()
