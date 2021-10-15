import os
import sys
import argparse
from pathlib import Path
import shutil

import numpy as np
import dlib


from mtcnn import MTCNN
import cv2


import imutils
from imutils import face_utils
import matplotlib.pyplot as plt
from skimage import io

#define the path setting arg
def dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path")

#set up argparse
parser = argparse.ArgumentParser()
#parser.add_argument("-i", dest="input_image", required="TRUE", type=argparse.FileType('r'))
parser.add_argument('-path', dest="input_path", required="TRUE", type=dir_path)
parser.add_argument("-d", dest="dataset_name")
args = parser.parse_args()

for entry in os.scandir(args.input_path):
    if (entry.path.endswith(".jpg") 
        or entry.path.endswith(".png")) and entry.is_file():
            #print(entry.path)
            
            #set input vars
            input_image_path = entry.path
            input_image_file = entry.name

            #set dataset var based on optional arg
            if args.dataset_name is not None:
                dataset_name = args.dataset_name
            else:
                dataset_name = Path(input_image_path).stem
    else:
        continue
print(dataset_name)
os._exit()





#set output vars
output_path = os.path.join('../datasets/', dataset_name)
if not os.path.exists(output_path):
    os.makedirs(output_path, exist_ok=True)
output_detections_path = os.path.join(output_path,'detections')
if not os.path.exists(output_detections_path):
    os.makedirs(output_detections_path, exist_ok=True)
output_detections_file = os.path.join(output_detections_path,Path(input_image_path).stem+'.txt')

#load the image and detect landmarks
img = cv2.cvtColor(cv2.imread(input_image_path), cv2.COLOR_BGR2RGB)
detector = MTCNN()
result = detector.detect_faces(img)
keypoints = result[0]['keypoints']
    
#copy the image to the dataset folder
shutil.copy2(input_image_path,output_path)

#write the landmarks to the detections folder in the dataset folder
with open(output_detections_file,'w') as f:
    f.write(str(keypoints['left_eye'][0])+' '+str(keypoints['left_eye'][1])+'\n')
    f.write(str(keypoints['right_eye'][0])+' '+str(keypoints['right_eye'][1])+'\n')
    f.write(str(keypoints['nose'][0])+' '+str(keypoints['nose'][1])+'\n')
    f.write(str(keypoints['mouth_left'][0])+' '+str(keypoints['mouth_left'][1])+'\n')
    f.write(str(keypoints['mouth_right'][0])+' '+str(keypoints['mouth_right'][1])+'\n')

