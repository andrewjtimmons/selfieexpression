import numpy as np
import pandas as pd
import cv2
from sklearn import datasets
import os

from PIL import Image
import urllib2
import cStringIO

FACE_CASECADE = cv2.CascadeClassifier('/Users/andrewjtimmons/anaconda/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
EYE_CASECADE = cv2.CascadeClassifier('/Users/andrewjtimmons/anaconda/share/OpenCV/haarcascades/haarcascade_eye.xml')
MOUTH_CASECADE = cv2.CascadeClassifier('/Users/andrewjtimmons/anaconda/share/OpenCV/haarcascades/haarcascade_mcs_mouth.xml')
SMILE_CASECADE = cv2.CascadeClassifier('/Users/andrewjtimmons/anaconda/share/OpenCV/haarcascades/haarcascade_smile.xml')

class Img():
    # Object for each image.

    def __init__(self, url):
        self.url = url
        self.color_image = self._create_opencv_image_from_url()
        self.grayscale_image = self._create_grayscale_image()
        self.faces_rois, self.faces = self._detect_faces()
        self.num_faces = len(self.faces_rois)
    
    def _create_opencv_image_from_url(self, cv2_img_flag = 1):
        # Get image from URL and convert to an openCV image.
        request = urllib2.urlopen(self.url)
        img_array = np.asarray(bytearray(request.read()), dtype=np.uint8)
        return cv2.imdecode(img_array, cv2_img_flag)
    
    def _create_grayscale_image(self):
        # Turn color image into grayscale.
        return cv2.cvtColor(self.color_image, cv2.COLOR_BGR2GRAY)
    
    def _detect_faces(self):
        # Detect faces in the image.  Returns an empty list if no faces.
        faces = FACE_CASECADE.detectMultiScale(self.grayscale_image, scaleFactor = 1.05, minNeighbors = 5)
        faces_rois = []
        for (x,y,w,h) in faces:
            self._draw_rectangle(self.color_image, (x,y), (x+w,y+h), (255,0,0))
            faces_rois.append(self.color_image[y:y+h, x:x+w])
        return faces_rois, faces
    
    def _draw_rectangle(self, image, pt1, pt2, color, thickness = 2):
        # Draw rectangle around a region of interests with a arbitrary color.
        cv2.rectangle(image, pt1, pt2, color, thickness)
        
    def show_color_image(self):
        # Display image on screen and close on key press.
        cv2.imshow('img',self.color_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def show_grayscale_image(self):
        # Display image on screen and close on key press.
        cv2.imshow('img',self.grayscale_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    def has_faces(self):
        # Returns True if more than one face is detected.
        if len(self.faces_rois) > 0:
            return True
        return False
        
        
class Face():
    # Object for a detected face region of interest.  Each image has 0 or more faces.
    
    def __init__(self, url, color_image, grayscale_image, face_roi, face_xywh):
        self.url = url
        self.color_image = color_image
        self.grayscale_image = grayscale_image
        self.face_roi = face_roi
        self.face_xywh = face_xywh
        self.eyes_rois, self.eyes_xywh_relative, self.eyes_xywh_absolute = self._detect_eyes()
        if self.has_two_eyes():
            self.roi_gray_below_eyes, self.roi_color_below_eyes = self._create_roi_below_eyes()
            self.mouth_rois, self.mouth_xywh_relative, self.mouth_xywh_absolute = self._detect_mouths()
            self.smile_rois, self.smile_xywh_relative, self.smile_xywh_absolute = self._detect_smiles()
        
    def _detect_eyes(self):
        # Detect eyes in the image.  Returns an empty list if no eyes.
        eyes_xywh_relative = EYE_CASECADE.detectMultiScale(self.face_roi, scaleFactor = 1.05, minNeighbors = 6)
        eyes_rois = []
        eyes_xywh_absolute = []
        for x, y, w, h in eyes_xywh_relative:
            self._draw_rectangle(self.face_roi, (x,y), (x+w,y+h), (0,255,0))
            x1, x2, y1, y2 = self._eye_math(x,y,w,h)
            self._draw_rectangle(self.color_image, (x1,y1), (x2,y2), (255,0,255), thickness = 1)
            eyes_rois.append(self.color_image[y1:y2, x1:x2])
            eyes_xywh_absolute.append([x1, x2, y1, y2])
        return eyes_rois, eyes_xywh_relative, eyes_xywh_absolute
    
    def _create_roi_below_eyes(self):
        # Takes the face roi and limits it to the region below the eyes.  This will let
        # the mouth casecades just search in that region instead of looking at the whole face.
        # Get x,y coords with width and height.
        x_face, y_face, w_face, h_face = self.face_xywh
        y_eyes = y_face + self.eyes_xywh_relative[0][1] + self.eyes_xywh_relative[0][3]
        face_bottom = h_face - self.eyes_xywh_relative[0][3]
        roi_gray_below_eyes = self.grayscale_image[y_eyes:y_eyes+face_bottom, x_face:x_face+w_face]
        roi_color_below_eyes = self.color_image[y_eyes:y_eyes+face_bottom, x_face:x_face+w_face]
        return roi_gray_below_eyes, roi_color_below_eyes
    
    def _detect_mouths(self):
        # Detect mouth in the image.  Returns an empty list if no mouth.
        mouth_xywh_relative = MOUTH_CASECADE.detectMultiScale(self.roi_gray_below_eyes, scaleFactor = 1.05, minNeighbors = 20)
        mouth_rois = []
        mouth_xywh_absolute = []
        for x,y,w,h in mouth_xywh_relative:
            self._draw_rectangle(self.roi_color_below_eyes, (x,y), (x+w,y+h), (0,0,0))
            x1, x2, y1, y2 = self._mouth_math(x,y,w,h)
            self._draw_rectangle(self.color_image, (x1,y1), (x2,y2), (255,255,0), thickness = 1)
            mouth_rois.append(self.color_image[y1:y2, x1:x2])
            mouth_xywh_absolute.append([x1, x2, y1, y2])
        return mouth_rois, mouth_xywh_relative, mouth_xywh_absolute
    
    def _detect_smiles(self):
        # Detect mouth in the image.  Returns an empty list if no mouth.
        smile_xywh_relative = SMILE_CASECADE.detectMultiScale(self.roi_gray_below_eyes, scaleFactor = 1.05, minNeighbors = 20)
        smile_rois = []
        smile_xywh_absolute = []
        for x , y, w, h in smile_xywh_relative:
            self._draw_rectangle(self.roi_color_below_eyes, (x,y), (x+w,y+h), (255,255,255))
            x1, x2, y1, y2 = self._mouth_math(x,y,w,h)
            self._draw_rectangle(self.color_image, (x1,y1), (x2,y2), (255,255,0), thickness = 1)
            smile_rois.append(self.color_image[y1:y2, x1:x2])
            smile_xywh_absolute.append([x1, x2, y1, y2])
        return smile_rois, smile_xywh_relative, smile_xywh_absolute
    
    def _eye_math(self, x, y, w, h):
        # Returns points from the eye roi that are in context of the whole image, not just the eye roi
        x1 = self.face_xywh[0]+x
        x2 = self.face_xywh[0]+x+w
        y1 = self.face_xywh[1]+y
        y2 = self.face_xywh[1]+y+h
        return x1, x2, y1, y2
    
    def _mouth_math(self, x, y, w, h):
        # Returns points from the mouth roi that are in context of the whole image, not just the mouth roi
        x1 = self.face_xywh[0]+x
        x2 = self.face_xywh[0]+x+w
        y1 = self.face_xywh[1]+self.eyes_xywh_relative[0][1]+self.eyes_xywh_relative[0][3]+y
        y2 = self.face_xywh[1]+self.eyes_xywh_relative[0][1]+self.eyes_xywh_relative[0][3]+y+h
        return x1, x2, y1, y2
    
    def _draw_rectangle(self, image, pt1, pt2, color, thickness = 2):
        # Draw rectangle around a region of interests with a arbitrary color.
        cv2.rectangle(image, pt1, pt2, color, thickness)
    
    def show_color_image(self):
        # Display image on screen and close on key press.
        cv2.imshow('img',self.color_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def show_grayscale_image(self):
        # Display image on screen and close on key press.
        cv2.imshow('img',self.grayscale_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def has_two_eyes(self):
        # Returns True if more than one face is detected.
        if len(self.eyes_rois) == 2:
            return True
        return False    
    

