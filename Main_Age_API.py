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
import imageio
from mlxtend.image import extract_face_landmarks
import creatww
from creatww import creatwindow

root = Tk()  # pour créer une nouvelle fénetre"
root.title("How old am I ?")
root.geometry("1112x540")
root.maxsize(1114, 548)
root.maxsize(1114, 548)
root.iconbitmap("image-removebg-preview (6).ico")
# Add image file
bg = PhotoImage(file="INETR1.png")
# Show image using label
label1 = Label(root, image=bg)
label1.place(x=0, y=0)
# create brows Button
bnt = Button(text="Start", font=("Courrier", 18), activebackground='red', command=lambda : creatwindow(root), bd=2, relief=SUNKEN)
#bnt = Button(text="Start", font=("Courrier", 18), activebackground='red',bd=2, relief=SUNKEN)
bnt.place(x=520, y=480)

root.mainloop()
