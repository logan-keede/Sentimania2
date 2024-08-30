
# align_faces.py

# import the necessary packages

from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
from imutils import paths
import argparse
import imutils
import dlib
import cv2
import os


# construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-p", "--shape-predictor", required=True,
#                 help="path to facial landmark predictor")
# ap.add_argument("-i", "--input", required=True,
# 	            help="path to input image")
# args = vars(ap.parse_args())
#
# if args == None:
#     raise Exception("could not load image !")


# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor and the face aligner




detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
fa = FaceAligner(predictor, desiredFaceWidth=48, desiredFaceHeight=48)


print("Image pre-processing is starting. Aligning image according to facial landmarks.")
# loop over the input images
for dataset_path in ["train", "test"]:
    for label in os.listdir(dataset_path):
        if os.path.isdir(os.path.join(dataset_path, label)):
            for filename in os.listdir(os.path.join(dataset_path, label)):
                if filename.endswith(".jpg") or filename.endswith(".png"):

                    inputPath = os.path.join(dataset_path, label, filename)
                    image = cv2.imread(inputPath)

                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                    # show the original input image and detect faces in the grayscale image
                    # rects = detector(gray, 2)
                    # print(rects)
                    # loop over the face detections
                    # for rect in rects:
                        # extract the ROI of the *original* face, then align the face
                        # using facial landmarks
                    print("path = ", inputPath)
                    rect = dlib.rectangle(left=0, top=48, right=48, bottom=0)
                    faceAligned = fa.align(image, gray, rect)
                    # print(inputPath.split("/")[-1])
                    # print(inputPath.split("/")[-2])

                    # write the output image to disk
                    path = 'new'
                    output_filename = os.path.join(path, label, filename)
                    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
                    cv2.imwrite(output_filename, faceAligned)
                    print(os.path.dirname(output_filename), output_filename)
                    # display the output images
                    cv2.imshow("Aligned", faceAligned)

                    cv2.waitKey(1)

print("Image face alignment is completed.")



# Command:
# python3 align_faces.py --shape-predictor dataset/shape_predictor_68_face_landmarks.dat --input dataset/input

# Reference:
# https://www.pyimagesearch.com/2017/05/22/face-alignment-with-opencv-and-python/