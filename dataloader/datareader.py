# -*- coding: utf-8 -*-
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


# Use only 3 classes.
# labels = ['background', 'person', 'car', 'road']


def pre_processing(img):
    ''' Random exposure and saturation (0.9 ~ 1.1 scale)'''
    
    rand_s = random.uniform(0.9, 1.1)
    rand_v = random.uniform(0.9, 1.1)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    tmp = np.ones_like(img[:, :, 1]) * 255
    img[:, :, 1] = np.where(img[:, :, 1] * rand_s > 255, tmp, img[:, :, 1] * rand_s)
    img[:, :, 2] = np.where(img[:, :, 2] * rand_v > 255, tmp, img[:, :, 2] * rand_v)

    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)

    # Centering helps normalization image (-1 ~ 1 value)
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
        print("Data_generator function should get mode arg 'train' or 'val' or 'test'.")
        return -1

    return x_data_gen_args, y_data_gen_args



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



def data_generator(d_path, b_size, mode):
    '''Data generator for fit_generator'''
    
    data = h5py.File(d_path, 'r')
    x_imgs = data.get('/' + mode + '/x')
    y_imgs = data.get('/' + mode + '/y')

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

            x.append(x_imgs[idx].reshape((256, 512, 3)))
            y.append(y_imgs[idx].reshape((256, 512, 1)))

            if len(x) == b_size:
                # Adapt ImageDataGenerator flow method for data augmentation.
                _ = np.zeros(b_size)
                seed = random.randrange(1, 1000)

                x_tmp_gen = x_data_gen.flow(np.array(x), _,
                                            batch_size=b_size,
                                            seed=seed)
                y_tmp_gen = y_data_gen.flow(np.array(y), _,
                                            batch_size=b_size,
                                            seed=seed)

                # Finally, yield x, y data.
                x_result, _ = next(x_tmp_gen)
                y_result, _ = next(y_tmp_gen)

                yield x_result, get_result_map(b_size, y_result)

                x.clear()
                y.clear()




if __name__ == "__main__":
    label,feature = read_data(PATH)


def showimage(img):
    #img = cv2.imread(feature[0],-1)
    #img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    cv2.imshow("window_name", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()