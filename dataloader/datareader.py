# -*- coding: utf-8 -*-
import numpy as np
import sys
sys.path.append('../')
import os
import random
import cv2
import matplotlib.pyplot as plt
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator



def read_image(path):
    
    img_path_labels_list=[os.path.join(path, img) for img in os.listdir(path) if img.endswith("_L.png")]
    img_path_features_list=[feature for feature in [os.path.join(path, img) for img in os.listdir(path)] if feature not in img_path_labels_list and feature.endswith(".png")] 
    
    img_path_labels_list = sorted(img_path_labels_list)
    img_path_features_list = sorted(img_path_features_list)
    
    feature = np.array([cv2.resize(img,(256,128) , interpolation = cv2.INTER_AREA) for img in
               [cv2.imread(i) for i in img_path_features_list]])
    label = np.array([cv2.resize(img,(256,128) , interpolation = cv2.INTER_AREA) for img in
             [cv2.imread(l) for l in img_path_labels_list]])
    return feature,label



def pre_processing(img):
    ''' Random exposure and saturation (0.9 ~ 1.1 scale)'''
    
    rand_s = random.uniform(0.9, 1.1)
    rand_v = random.uniform(0.9, 1.1)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    tmp = np.ones_like(img[:, :, 1]) * 255
    img[:, :, 1] = np.where(img[:, :, 1] * rand_s > 255, tmp, img[:, :, 1] * rand_s)
    img[:, :, 2] = np.where(img[:, :, 2] * rand_v > 255, tmp, img[:, :, 2] * rand_v)

    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)

    # normalization of image 
    return img / 127.5 - 1



def get_data_gen_args(mode):
    ''' Get ImageDataGenerator arguments(options) depends on mode - (train, val, test)'''
    
    if mode == 'train' or mode == 'val':
        x_data_gen_args = dict(preprocessing_function=pre_processing,
                               shear_range=0.1,
                               zoom_range=0.1,
                               rotation_range=10,
                               width_shift_range=0.1,
                               height_shift_range=0.1,
                               fill_mode='constant',
                               horizontal_flip=True)

        y_data_gen_args = dict(shear_range=0.1,
                               zoom_range=0.1,
                               rotation_range=10,
                               width_shift_range=0.1,
                               height_shift_range=0.1,
                               fill_mode='constant',
                               horizontal_flip=True)

    elif mode == 'test':
        x_data_gen_args = dict(preprocessing_function=pre_processing)
        y_data_gen_args = dict()

    else:
        print("Invalid mode only 'train' or 'val' or 'test' allowed")
        return -1

    return x_data_gen_args, y_data_gen_args



def get_result_map(batch_size, label_img):
    '''One hot encoding for label_img.'''
    
    #label_img = np.squeeze(label_img, axis=-1)
    result_map = np.zeros((batch_size, 128, 256, 4),dtype = np.uint8)
    
    # For np.where calculation.
    building = (label_img == [0 ,0, 128]).all(axis = -1)  #bgr
    car = (label_img == [128,0,64]).all(axis = -1)  #BGR
    road = (label_img == [128,64,128]).all(axis = -1)
    background = np.logical_not(building + car + road)

    result_map[:, :, :, 0] = np.where(building, 1, 0)
    result_map[:, :, :, 1] = np.where(car, 1, 0)
    result_map[:, :, :, 2] = np.where(road, 1, 0)
    result_map[:, :, :, 3] = np.where(background, 1, 0)

    return result_map



def data_generator(dir_path, batch_size, mode):
    '''Data generator for fit_generator'''
    
    
    x_imgs,y_imgs = read_image(dir_path)
    a = y_imgs.shape
    print(f"**********************{a}***************")
     
    # Make ImageDataGenerator.
    x_data_gen_args, y_data_gen_args = get_data_gen_args(mode)
    x_data_gen = ImageDataGenerator(**x_data_gen_args)
    y_data_gen = ImageDataGenerator(**y_data_gen_args)

    # random index for random data access.
    d_size = x_imgs.shape[0]
    shuffled_idx = list(range(d_size))

    x = []
    y = []
    while True:
        random.shuffle(shuffled_idx)
        for i in range(d_size):
            idx = shuffled_idx[i]

            x.append(x_imgs[idx].reshape((128, 256, 3)))
            y.append(y_imgs[idx].reshape((128, 256, 3)))

            if len(x) == batch_size:
                # Adapt ImageDataGenerator flow method for data augmentation.
                _ = np.zeros(batch_size)
                seed = random.randrange(1, 1000)

                x_tmp_gen = x_data_gen.flow(np.array(x), _,
                                            batch_size=batch_size,
                                            seed=seed)
                y_tmp_gen = y_data_gen.flow(np.array(y), _,
                                            batch_size=batch_size,
                                            seed=seed)

                # Finally, yield x, y data.
                x_result, _ = next(x_tmp_gen)
                y_result, _ = next(y_tmp_gen)

                yield x_result, get_result_map(batch_size, y_result)

                x.clear()
                y.clear()


         

if __name__ == "__main__":
    feature,label = read_image(DATASET_PATH)