"""Looks to see when a face angle turned too far away to be seen.  You can 
make your own images with a protractor and a rope.  Measure to the wall and mark 
with tape.  Then look at the spot and take a photo with your webcam."""

import numpy as np
import cv2
import os

face_cascade = cv2.CascadeClassifier('/Users/andrewjtimmons/anaconda/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('/Users/andrewjtimmons/anaconda/share/OpenCV/haarcascades/haarcascade_eye.xml')
mouth_cascade = cv2.CascadeClassifier('/Users/andrewjtimmons/anaconda/share/OpenCV/haarcascades/haarcascade_mcs_mouth.xml')
nose_cascade = cv2.CascadeClassifier('/Users/andrewjtimmons/anaconda/share/OpenCV/haarcascades/haarcascade_mcs_nose.xml')
smile_cascade = cv2.CascadeClassifier('/Users/andrewjtimmons/anaconda/share/OpenCV/haarcascades/haarcascade_smile.xml')


path = 'angle_tester/'
dirListing = os.listdir(path)
imgList = []
imgNameList = []
for item in dirListing:
  if ".jpg" in item:
    imgList.append(path+item)
    imgNameList.append(item)
print imgList

for image, name in zip(imgList, imgNameList):
    img = cv2.imread(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    scaleFactor = 1.05
    minNeighbors = 6
    faces = face_cascade.detectMultiScale(img, scaleFactor = scaleFactor, minNeighbors = minNeighbors)

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        # roi_gray = gray[y:y+h, x:x+w]
        # roi_color = img[y:y+h, x:x+w]
        # eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor = scaleFactor, minNeighbors = minNeighbors)
        # for (ex,ey,ew,eh) in eyes:
        #     cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    cv2.imwrite("angle_tested/" + name,img)

    
