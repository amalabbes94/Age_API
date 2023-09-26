from tkinter import *
from tkinter import messagebox
import cv2
from PIL import ImageTk, Image
from tkinter import filedialog
import os
import tkinter
import matplotlib.pyplot as plt


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
#from creatww import creatwindow
import cross_distances
from cross_distances import predict_age_range, calc_cross_distances



class App:
        global name
        def __init__(self, window, window_title, video_source=0):
            self.window = window
            self.window.title(window_title)
            self.video_source = video_source
            self.ok = False


            # timer
            '''self.timer = ElapsedTimeClock(self.window)'''

            # open video source (by default this will try to open the computer webcam)
            self.vid = VideoCapture(self.video_source)

            # Create a canvas that can fit the above video source size
            self.canvas = tk.Canvas(window, width=640, height=480)
            self.canvas.pack()

            # Button that lets the user take a snapshot
            self.btn_snapshot = tk.Button(window, text="Save image", command=self.snapshot, font=("Courrier", 18),
                                          activebackground='red', bg='#FF5050', fg='white')
            self.btn_snapshot.place(x=740, y=580)
            self.btn_snapshot.pack()

            self.delay = 10
            self.update()


            self.window.mainloop()

        def snapshot(self):
            # Get a frame from the video source
            ret, frame = self.vid.get_frame()
            global name
            if ret:
                name = time.strftime("%d-%m-%Y-%H-%M-%S") + ".png"

                cv2.imwrite("frame-" + name, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                self.window.destroy()
                img2 = ImageTk.PhotoImage(Image.open("frame-"+name))

                label2.configure(image=img2)
                label2.image = img2

        def update(self):

            # Get a frame from the video source
            ret, frame = self.vid.get_frame()
            if self.ok:
                self.vid.out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            if ret:
                self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
                self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
            self.window.after(self.delay, self.update)

class VideoCapture:
        def __init__(self, video_source=0):
            # Open the video source
            self.vid = cv2.VideoCapture(video_source)
            if not self.vid.isOpened():
                raise ValueError("Unable to open video source", video_source)

        # To get frames
        def get_frame(self):
            if self.vid.isOpened():
                ret, frame = self.vid.read()

                if ret:
                    # Return a boolean success flag and the current frame converted to BGR
                    return (ret, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                else:
                    return (ret, None)
            else:
                return (ret, None)



def main():
    global label2, root2, name
    root3 = tkinter.Toplevel(root2)  # pour créer une nouvelle fénetre"
    root3.title("How old am I ?")
    root3.geometry("1112x540")
    root3.maxsize(600, 544)
    App(root3, 'Video Recorder')
    root3.mainloop()


def showlandmarks():
    global name
    #imm = imageio.imread("frame-"+name)
    '''landmarks = extract_face_landmarks(imm)
    for p in landmarks:
        imm[p[1] - 3:p[1] + 3, p[0] - 3:p[0] + 3, :] = (255, 0, 255)'''


    predictor_path = "C:/Users/amal/PycharmProjects/pythonProject/shape_predictor_68_face_landmarks.dat"  # Replace with the path to your model file
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    # Load an image using OpenCV (replace 'image.jpg' with your image file's path)
    image = imageio.imread("frame-"+name)

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

            # Draw a circle at each landmark point (you can customize the radius and color)
            cv2.circle(image, (x, y), 3, (0, 255, 0), -1)
            cv2.imwrite("imam.jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            imm = ImageTk.PhotoImage(Image.open("imam.jpg"))

            label3.configure(image=imm)
            label3.image = imm

    # Display or save the image with facial landmarks
    # cv2.imshow("Image with Facial Landmarks", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    age_features = calc_cross_distances(landmarks, 0)
    result = predict_age_range(age_features)

    label4 = Label(root2, text="Your age range is" + str(result), fg='red', font=("Courrier", 18))
    label4.place(x=450, y=500)



'''def show_prediction_result():
    landmarks=showlandmarks()
    age_features = calc_cross_distances(landmarks,0)
    age_range=predict_age_range(age_features)
    # Display or save the image with facial landmarks
    #cv2.imshow("Image with Facial Landmarks", image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return age_range'''






def creatwindow(root):
    global root2, label2,label3, name
    root2 = tkinter.Toplevel(root)   # pour créer une nouvelle fénetre"
    root2.title("How old am I ?")
    root2.geometry("1112x540")
    root2.maxsize(1110, 537)
    root2.maxsize(1110, 537)
    root2.iconbitmap("image-removebg-preview (6).ico")
    # Add image file
    bg = PhotoImage(file="Int22.png")
    # Show image using label
    label1 = Label(root2, image=bg)
    label1.place(x=0, y=0)

    '''img = Image.open('C:/Users/amal/PycharmProjects/pythonProject/fingerprint3.png')
    img.thumbnail((310, 330))
    img = ImageTk.PhotoImage(img)

    label2 = Label(root2,width=290, height=300,image=img)
    label2.place(x=220, y=80)'''

    img = Image.open('C:/Users/amal/PycharmProjects/pythonProject/im1in2.png')
    img.thumbnail((318, 187))
    img = ImageTk.PhotoImage(img)

    label2 = Label(root2, width=318, height=187, image=img)
    label2.place(x=200, y=220)





    bnt1=Button(root2,text="Select your image",font=("Courrier", 18),activebackground='red',command=main,bd=2,relief=SUNKEN,bg='#0A0444',fg='white')
    bnt1.place(x=250, y=410)

    img2 = Image.open('C:/Users/amal/PycharmProjects/pythonProject/im2in2.png')
    img2.thumbnail((321, 188))
    img2 = ImageTk.PhotoImage(img2)
    label3 = Label(root2, width=321, height=188, image=img2)
    label3.place(x=650, y=220)
    bnt2 = Button(root2, text="Show landmarks", font=("Courrier", 18), activebackground='red',command=showlandmarks,
              bd=2, relief=SUNKEN, bg='#FF5050', fg='white')
    bnt2.place(x=700, y=410)


    root2.mainloop()