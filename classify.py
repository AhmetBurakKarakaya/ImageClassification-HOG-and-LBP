# -*- coding: utf-8 -*-
"""
Created on Wed May 25 20:58:50 2022

@author: karak
"""
import os
import cv2
import argparse
from sklearn.svm import LinearSVC
from skimage import feature
from sklearn.neighbors import KNeighborsClassifier
from skimage import exposure

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import skimage.io, skimage.color
import HOG
import lbp as l
from PIL import ImageTk, Image

data = []
labels = []


# get all the image folder paths
outPutPath = os.listdir("yalefaces")



for imName in outPutPath:
      if imName.endswith(".gif"):
          make = imName.split("/")[-1].split(".")[0].replace('subject', '') + imName.split("/")[-1].split(".")[1]
          
          img = skimage.io.imread("yalefaces"+"/"+imName)
          """
          height, width = img.shape
          
          img_lbp = np.zeros((height, width), np.uint8)
          for i in range(0, height):
              for j in range(0, width):
                   img_lbp[i, j] = l.lbp_calculated_pixel(img, i, j)
          """
          # Hog implement

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
          H = HOG.HOG_cell_histogram(cell_direction, cell_magnitude, hist_bins)
          
          data.append(H)
          labels.append(make)
          

index = -1
if "05sleepy" in labels:
    #88
    index = labels.index("05sleepy")
    


model = KNeighborsClassifier(n_neighbors=1)
model.fit(data, labels)

img = skimage.io.imread("yalefaces/subject05.sleepy.gif")

# predict
pred1 = model.predict(data)
pred = model.predict(data[index].reshape(1, -1))[0]

from sklearn.metrics import accuracy_score
# printing accuracy
print(accuracy_score(labels,pred1))



from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.svm import SVC

cm = confusion_matrix(labels, pred1)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()

plt.show()





hogImage = exposure.rescale_intensity(img, out_range=(0, 255))
hogImage = hogImage.astype("uint8")
cv2.imshow("HOG Image #{}".format(1), hogImage)

# draw the prediction on the test image and display it
cv2.putText(img, pred.title(), (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0,(0, 255, 0), 3)
cv2.imshow("Test Image #{}".format(1), img)
cv2.waitKey(0)















