import os
import sys
from mtcnn import MTCNN
import cv2

img = cv2.cvtColor(cv2.imread("000002.jpg"), cv2.COLOR_BGR2RGB)
detector = MTCNN()
result = detector.detect_faces(img)
keypoints = result[0]['keypoints']
print(keypoints['left_eye'][0])
print(keypoints['right_eye'])
print(keypoints['nose'])
print(keypoints['mouth_left'])
print(keypoints['mouth_right'])

with open('000002.txt','w') as f:
    f.write(str(keypoints['left_eye'][0])+' '+str(keypoints['left_eye'][1])+'\n')
