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

predictor_path = "shape_predictor_68_face_landmarks.dat"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
img = dlib.load_rgb_image('inputimage.jpg')

rect = detector(img)[0]
sp = predictor(img, rect)
landmarks = np.array([[p.x, p.y] for p in sp.parts()])

print(landmarks)
for x in landmarks:
print(x)
    
f = open( 'inputimage.txt', 'w' )
#f.write( landmarks )
f.close()