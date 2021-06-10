# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 16:15:10 2021

@author: charu
"""

import numpy as np
import os
import random
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from config.config import PATH


def read_data(path):
    img_path_labels_list=[os.path.join(path, img) for img in os.listdir(path) if img.endswith("_L.png")]
    img_path_features_list=[feature for feature in [os.path.join(path, img) for img in os.listdir(path)] 
                            if feature not in img_path_labels_list and feature.endswith(".png")] 
    return img_path_labels_list,img_path_features_list



def get_result_map(b_size, y_img):
    '''One hot encoding for y_img.'''
    
    #y_img = np.squeeze(y_img, axis=-1)
    result_map = np.zeros((b_size, 256, 512, 3),dtype = np.int8)

    # For np.where calculation.
    person = (y_img == [0,0,0]).all(axis = 2)
    car = (y_img == [64,0,128]).all(axis = 2)
    road = (y_img == [128,64,128]).all(axis = 2)
    background = np.logical_not(person + car + road)

    result_map[:, :, :, 0] = np.where(background, 1, 0)
    result_map[:, :, :, 1] = np.where(person, 1, 0)
    result_map[:, :, :, 2] = np.where(road, 1, 0)
    #result_map[:, :, :, 3] = np.where(car, 1, 0)

    return result_map


def showimage(img):
    #img = cv2.imread(feature[0],-1)
    #img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    cv2.imshow("window_name", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
if __name__ == "__main__":
    label,feature = read_data(PATH)