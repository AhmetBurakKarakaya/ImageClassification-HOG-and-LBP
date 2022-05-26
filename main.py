# -*- coding: utf-8 -*-
"""
Created on Fri May 13 14:38:43 2022

@author: karak
"""

import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox

from PIL import ImageTk, Image

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import skimage.io, skimage.color
import HOG
import lbp as l
from pylab import plot, show, xlabel, ylabel

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, 
NavigationToolbar2Tk)






window = tk.Tk()
window.geometry("1080x640")
window.wm_title("LBP - HOG Classification")

#Global variables
img_name = ""
count = 0
expected_output = ""


#Frames
frame_left = tk.Frame(window, width= 540, height=640, bd= "2")
frame_left.grid(row=0, column=0)

frame_right = tk.Frame(window, width= 540, height=640, bd= "2")
frame_right.grid(row=0, column=1)


frame1 = tk.LabelFrame(frame_left, text = "Image", width=540, height = 500)
frame1.grid(row = 0, column = 0)


frame2 = tk.LabelFrame(frame_left, text = "Model and Save", width=540, height = 140)
frame2.grid(row = 1, column = 0)


frame3 = tk.LabelFrame(frame_right, text = "Features", width=270, height = 640)
frame3.grid(row = 0, column = 0)


frame4 = tk.LabelFrame(frame_right, text = "Result", width=270, height = 640)
frame4.grid(row = 0, column = 1, padx = 10)


#frame1

def imageResize(img):
    basewidth = 500
    wpercent = (basewidth / float (img.size[0]))
    hsize = int( (float(img.size[1])* float(wpercent)))
    img = img.resize((basewidth, hsize), Image.ANTIALIAS)
    return img
    
    
def openImage():
    global img_name
    global count
    global expected_output
    
    count+=1
    if count != 1:
        messagebox.showinfo(title="Warning", message = "Only one image can be opened")
    else:
        img_name = filedialog.askopenfilename(initialdir = "C:\\Users\\karak\\Desktop\\Görüntü Proje", title = "Select an image file")
        img_jpg = img_name.split("/")[-1].split(".")[0]
        expected_output = img_name.split("/")[-1].split(".")[0].replace('subject', '') + img_name.split("/")[-1].split(".")[1]
        tk.Label(frame1, text = img_jpg, bd = 3).pack(pady = 10)
        tk.Label(frame3, font = ("Times", 12), text = "expected_output : " + expected_output ).place(relx = x, rely = y)
        
        #open and show
        
        img = Image.open(img_name)
        img = imageResize(img)
        img = ImageTk.PhotoImage(img)
        panel = tk.Label(frame1, image = img)
        panel.image = img
        panel.pack(padx = 15, pady = 15)
        
        


menubar = tk.Menu(window)
window.config(menu = menubar)
file = tk.Menu(menubar)
menubar.add_cascade(label = "File", menu = file)
file.add_command(label="Open", command = openImage)

#frame 3
def classification():
    
    if img_name != "" and models.get() != "":
        if models.get() == "HOG":
            # Hog implement

            img = skimage.io.imread(img_name)
            #img = skimage.color.rgb2gray("subject01.jpg")
            horizontal_mask = np.array([-1, 0, 1])
            vertical_mask = np.array([[-1],
                                     [0],
                                     [1]])

            horizontal_gradient = HOG.calculate_gradient(img, horizontal_mask)
            vertical_gradient = HOG.calculate_gradient(img, vertical_mask)

            grad_magnitude = HOG.gradient_magnitude(horizontal_gradient, vertical_gradient)
            grad_direction = HOG.gradient_direction(horizontal_gradient, vertical_gradient)

            grad_direction = grad_direction % 180
            hist_bins = np.array([10,30,50,70,90,110,130,150,170])

            
            cell_direction = grad_direction[:8, :8]
            cell_magnitude = grad_magnitude[:8, :8]
            HOG_cell_hist = HOG.HOG_cell_histogram(cell_direction, cell_magnitude, hist_bins)
        
            plt.bar(x=np.arange(9), height=HOG_cell_hist, align="center", width=0.8)
            plt.show()
            
            # the figure that will contain the plot
            fig = Figure(figsize = (5, 5),
                 dpi = 100)
            
            data = HOG_cell_hist
            # adding the subplot
            plot1 = fig.add_subplot(111)
            plot1.plot(data.tolist())
            canvas = FigureCanvasTkAgg(fig,
                               master = frame4)  
            canvas.draw()
            canvas.get_tk_widget().pack()
            # creating the Matplotlib toolbar
            toolbar = NavigationToolbar2Tk(canvas,
                                   window)
            toolbar.update()
  
            # placing the toolbar on the Tkinter window
            canvas.get_tk_widget().pack()
            
            
        
        if models.get() == "LBP":
            # Hog implement

            img = skimage.io.imread(img_name)
            #img = skimage.color.rgb2gray("subject01.jpg")
            
            height, width = img.shape
            
            img_lbp = np.zeros((height, width), np.uint8)
            for i in range(0, height):
                for j in range(0, width):
                     img_lbp[i, j] = l.lbp_calculated_pixel(img, i, j)
            
            
            horizontal_mask = np.array([-1, 0, 1])
            vertical_mask = np.array([[-1],
                                     [0],
                                     [1]])

            horizontal_gradient = HOG.calculate_gradient(img_lbp, horizontal_mask)
            vertical_gradient = HOG.calculate_gradient(img_lbp, vertical_mask)

            grad_magnitude = HOG.gradient_magnitude(horizontal_gradient, vertical_gradient)
            grad_direction = HOG.gradient_direction(horizontal_gradient, vertical_gradient)

            grad_direction = grad_direction % 180
            hist_bins = np.array([10,30,50,70,90,110,130,150,170])

            # Histogram of the first cell in the first block.
            cell_direction = grad_direction[:8, :8]
            cell_magnitude = grad_magnitude[:8, :8]
            HOG_cell_hist = HOG.HOG_cell_histogram(cell_direction, cell_magnitude, hist_bins)
        
            plt.bar(x=np.arange(9), height=HOG_cell_hist, align="center", width=0.8)
            print(HOG_cell_hist.shape)
            plt.show()
            
            # the figure that will contain the plot
            fig = Figure(figsize = (5, 5),
                 dpi = 100)
            
            data = HOG_cell_hist
            # adding the subplot
            plot1 = fig.add_subplot(111)
            plot1.plot(data.tolist())
            canvas = FigureCanvasTkAgg(fig,
                               master = frame4)  
            canvas.draw()
            canvas.get_tk_widget().pack()
            # creating the Matplotlib toolbar
            toolbar = NavigationToolbar2Tk(canvas,
                                   window)
            toolbar.update()
  
            # placing the toolbar on the Tkinter window
            canvas.get_tk_widget().pack()
            

x = 0.1
y = (1/10) / 2


classify_btn = tk.Button(frame3, bg = "red", bd = 4, font = ("Times", 13), activebackground = "orange", text = "Classify", command = classification)
classify_btn.place(relx = 0.1, rely = 0.5)

##





## frame2

#combo box
model_selection_label = tk.Label(frame2, text = "Choose classification model : ")
model_selection_label.grid(row=0, column = 0, padx = 5)

models = tk.StringVar()
model_selection = ttk.Combobox(frame2, textvariable = models, values = ("LBP", "HOG"), state = "readonly")
model_selection.grid(row=0, column = 1, padx = 5)

# check box

chvar = tk.IntVar()
chvar.set(0)
xbox = tk.Checkbutton(frame2, text = "Save Classification Result", variable = chvar)
xbox.grid(row=1, column = 0, pady = 5)

#entry
entry = tk.Entry(frame2, width = 23)
entry.insert(index = 0, string = "Saving name...")
entry.grid(row=1, column = 1)



























window.mainloop()















