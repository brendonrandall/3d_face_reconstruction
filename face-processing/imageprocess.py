import os
import sys
import argparse
from pathlib import Path
from shutil import copyfile

import numpy as np
import dlib
import imutils
from imutils import face_utils
import matplotlib.pyplot as plt
from skimage import io

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", dest="input_image", required="TRUE", type=argparse.FileType('r'))
parser.add_argument("-d","--dataset",dest="dataset_name", required="TRUE")

p = parser.parse_args()

print(dlib.__version__)
print(imutils.__version__)

output_path = os.path.join('../datasets/', dataset_name)
if not os.path.exists(output_path):
    os.makedirs(output_path, exist_ok=True)
output_detections_path = os.path.join(output_path,'detections')
output_detections_file = os.path.join(output_detections_path,dataset_name,'.txt')

predictor_path = "shape_predictor_5_face_landmarks.dat"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
img = dlib.load_rgb_image(p.input_image)

rect = detector(img)[0]
sp = predictor(img, rect)
landmarks = np.array([[p.x, p.y] for p in sp.parts()])

print(landmarks)
#for x in landmarks:
    #print(x)
 
#f = open( 'inputimage.txt', 'w' )
#f.write( landmarks )
#f.close()

#np.savetxt("inputimage.txt",x)
mat = np.matrix(landmarks)
with open(output_detections_file,'wb') as f:
    for line in mat:
        np.savetxt(f, line, fmt='%.2f')


