import cv2
import time
import numpy as np


class HAARClassifier:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier('Classifiers/Cascades/haarcascade_frontalface_default.xml')
        self.fullbody_cascade = cv2.CascadeClassifier('Classifiers/Cascades/haarcascade_fullbody.xml')
        self.upperbody_cascade = cv2.CascadeClassifier('Classifiers/Cascades/haarcascade_upperbody.xml')
        self.lowerbody_cascade = cv2.CascadeClassifier('Classifiers/Cascades/haarcascade_lowerbody.xml')

    def detect_faces(self, frame):
        #self.faces = self.face_cascade.detectMultiScale(frame, 1.1, 3)
        self.faces = self.face_cascade.detectMultiScale(frame, 1.1, 4)
        #print "HOG CLASSIFIER DETECT RETURN"
        return self.faces

    def detect_body(self, frame):
        #self.fullbody = self.fullbody_cascade.detectMultiScale(frame, 1.1, 3)
        self.fullbody = self.fullbody_cascade.detectMultiScale(frame, 1.1, 4)
        #print "HOG CLASSIFIER DETECT RETURN"
        return self.fullbody

    def detect_lower(self, frame):
        self.lowerbody = self.lowerbody_cascade.detectMultiScale(frame, 1.1, 4)
        return self.lowerbody

    def detect_upper(self, frame):
        self.upperbody = self.upperbody_cascade.detectMultiScale(frame, 1.1, 4)
        return self.upperbody




