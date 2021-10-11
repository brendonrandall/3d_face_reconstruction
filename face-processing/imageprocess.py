import os
import sys
import argparse
from pathlib import Path
import numpy as np
import dlib
import imutils
from imutils import face_utils
import matplotlib.pyplot as plt
from skimage import io

parser = argparse.ArgumentParser()
#parser.add_argument("-f", "--file", dest="input_image", required="TRUE", type=argparse.FileType('r'))
#parser.add_argument("-p", "--predictor", dest="landmark_predictor", required="TRUE", type=argparse.FileType('r'))

#p = parser.parse_args()

print(dlib.__version__)
print(imutils.__version__)

predictor_path = "shape_predictor_5_face_landmarks.dat"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

img = io.imread("inputimage.jpg")

dets = detector(img)

#output face landmark points inside retangle
#shape is points datatype
#http://dlib.net/python/#dlib.point
for k, d in enumerate(dets):
    shape = predictor(img, d)

vec = np.empty([68, 2], dtype = int)
for b in range(68):
    vec[b][0] = shape.part(b).x
    vec[b][1] = shape.part(b).y

print(vec)