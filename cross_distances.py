from tkinter import *
from tkinter import messagebox
import cv2
from PIL import ImageTk, Image
from tkinter import filedialog
import os
import tkinter
import matplotlib.pyplot as plt


import sklearn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tkinter as tk
import cv2
import PIL.Image, PIL.ImageTk
import time
import datetime as dt
import argparse
import dlib
import cv2

import dlib
import imageio
from mlxtend.image import extract_face_landmarks
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance

name="frame-24-10-2022-21-43-24.png"
predictor_path = "C:/Users/amal/PycharmProjects/pythonProject/shape_predictor_68_face_landmarks.dat"  # Replace with the path to your model file
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

    # Load an image using OpenCV (replace 'image.jpg' with your image file's path)
image = imageio.imread(name)

    # Convert the image to grayscale for better detection
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
faces = detector(gray)

for face in faces:
        # Calculate facial landmarks for each detected face
        landmarks = predictor(gray, face)

        # Iterate through the 68 landmarks and access their x, y coordinates
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(image, (x, y), 3, (0, 255, 0), -1)

vec=np.zeros((1,135),np.float32)
max_block=np.zeros((1,4),np.float32)
ratios=np.zeros((1,10),np.float32)


def calc_cross_distances(landmarks,m):#,m,img):

    vec=np.zeros((1,135),np.float32)
    d=-1
    m1=-1


# the face boundry is descrbed by points indexed from 1 to 15
    for k1 in range(1,9):
                    x=landmarks.part(k1).x
                    y=landmarks.part(k1).y
                    for k2 in range(9,16):
                        x1=landmarks.part(k2).x
                        y1=landmarks.part(k2).y
                        cv2.line(image, (x,y),(x1,y1), (255, 255, 0))
                        m1=m1+1
                        d=d+1
                        #print('avant erreur',m,d)
                        vec[m,d]= distance.cosine((x,y),(x1,y1))


    max_block[0,0]=max(vec[m,0:d])


#the eyes block
    right_eyes = [22, 23, 24, 25, 26, 42, 43, 44, 45, 46, 47]
    left_eyes =  [17, 18, 19, 20, 21, 36, 37, 38, 39, 40, 41]
    m2=d
    #print(m2)
    for k1 in range(0, len(right_eyes)):
            x = landmarks.part(right_eyes[k1]).x
            y = landmarks.part(right_eyes[k1]).y
            x1 = landmarks.part(36).x
            y1 = landmarks.part(36).y
            cv2.line(image, (x, y), (x1, y1) ,(255, 0, 0))
            d = d + 1
            vec[m, d] = distance.cosine((x, y), (landmarks.part(36).x, landmarks.part(36).y))
    for k1 in range(0, len(left_eyes)):
            x = landmarks.part(left_eyes[k1]).x
            y = landmarks.part(left_eyes[k1]).y
            x1 = landmarks.part(45).x
            y1 = landmarks.part(45).y
            cv2.line(image, (x, y), (x1, y1) ,(255, 0, 0))
            d = d + 1
            vec[m, d] = distance.cosine((x, y), (landmarks.part(45).x, landmarks.part(45).y))


    d = d + 1
    vec[m, d] = distance.cosine((landmarks.part(36).x, landmarks.part(36).y), (landmarks.part(45).x, landmarks.part(45).y))
    max_block[0,1]= max(vec[0,m2+1:d])
    #print('block 2 max between:', m2,d)


# Noise block
    Noise1 = [27, 28, 29, 30]
    Noise2 = [31, 32, 33, 34, 35]
    m3=d
    for k1 in Noise1:
            x = landmarks.part(k1).x
            y = landmarks.part(k1).y
            for k2 in Noise2:
                x1 = landmarks.part(k2).x
                y1 = landmarks.part(k2).y
                d = d + 1
                cv2.line(image, (x,y), (x1, y1), (255, 0, 0))
                vec[m, d] = distance.cosine((x, y), (x1, y1))
   # print(d)
    max_block[0,2]= max(vec[m, m3+1:d])
    #print('block 3 max between:', m3,d)

# mouth block :
    m4=d
    m1= [48, 49, 50,  59, 58, 57]
    m2=[52, 53, 54,  56,57]
    for k1 in range(0, len(m1)):
       x = landmarks.part(m1[k1]).x
       y = landmarks.part(m1[k1]).y
       for k2 in range(0, len(m2)):
          x1 = landmarks.part(m2[k2]).x
          y1 = landmarks.part(m2[k2]).y
          cv2.line(image, (x, y), (x1, y1), (0, 255, 255))
          d = d + 1
          vec[m, d] = distance.cosine((x, y), (x1, y1))
    #print(d)
    max_block[0,3]= max(vec[m, m4+1:d])
    #print('block 4max between:', m4,d)

#calculates ratios of max distances
    k3=-1
    for k1 in range(0,4):
            for k2 in range(k1+1,4):
                k3=k3+1
                d=d+1
                ratios[0,k3]=np.float64(max_block[0, k1] / max_block[0, k2])
                vec[m, d] = np.float64(max_block[0, k1] / max_block[0, k2])
               # print("the ratio between ",max_block[0, k1] , ' and ',  max_block[0, k2], ' is', (np.float64(max_block[0, k1] / max_block[0, k2])))
    return vec
vec= calc_cross_distances(landmarks,0)

#print('the vector is',vec)
#print('the max bocks are ',max_block)
#print('the ratios are:', ratios)
#cv2.imwrite("crossdis_ima.png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


import joblib
def predict_age_range (vec):
   loaded_model = joblib.load('svm_level1.pkl')
   ypred = loaded_model.predict(vec)
   #print(ypred)
   if ypred ==0:
    child_model = joblib.load('svm_level2_child.pkl')
    decision = child_model.predict(vec)
   elif ypred == 1:
    adult_model = joblib.load('svm_level2_adult.pkl')
    decision = adult_model.predict(vec)
   else:
    adult_model = joblib.load('svm_level2_older.pkl')
    decision = adult_model.predict(vec)
#print(classification_report(labels_test,ypred))
#print(confusion_matrix(labels_test,ypred))

   if ypred==0 and decision==0:
      predicted_class="[0-2]"
   elif ypred==0  and  decision==1:
    predicted_class = "[4-6]"
   elif ypred == 0 and decision == 2:
    predicted_class = "[8-12]"

   elif ypred==1 and decision==0:
    predicted_class="[15-20]"
   elif ypred==0  and  decision==1:
    predicted_class = "[25-32]"
   elif ypred == 0 and decision == 2:
    predicted_class = "[38-43]"

   elif ypred==2 and decision==0:
    predicted_class="[48-53]"
   elif ypred==2  and  decision==1:
    predicted_class = "[60-100]"
   return predicted_class
#print("the predicted class is",predict_age_range(vec))

