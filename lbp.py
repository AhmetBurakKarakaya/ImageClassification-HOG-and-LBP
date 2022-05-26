import cv2
import numpy as np
from matplotlib import pyplot as plt

def get_pixel(img, center, x, y):
    new_value = 0
    try:
        if img[x][y] >= center:
            new_value = 1
    except:
        pass
    return new_value

def lbp_calculated_pixel(img, x, y):
    '''

     64 | 128 |   1
    ----------------
     32 |   0 |   2
    ----------------
     16 |   8 |   4    

    '''    
    center = img[x][y]
    val_ar = []
    val_ar.append(get_pixel(img, center, x-1, y+1))   
    val_ar.append(get_pixel(img, center, x, y+1))       
    val_ar.append(get_pixel(img, center, x+1, y+1))     
    val_ar.append(get_pixel(img, center, x+1, y))       
    val_ar.append(get_pixel(img, center, x+1, y-1))     
    val_ar.append(get_pixel(img, center, x, y-1))       
    val_ar.append(get_pixel(img, center, x-1, y-1))   
    val_ar.append(get_pixel(img, center, x-1, y))       
    
    power_val = [1, 2, 4, 8, 16, 32, 64, 128]
    val = 0
    for i in range(len(val_ar)):
        val += val_ar[i] * power_val[i]
    return val    
    
