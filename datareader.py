# -*- coding: utf-8 -*-
import numpy as np
import os

path = r"F:\DS\Dataset\Multi_class_image_segmentation\CamSeq01"


def read_data(path):
    img_path_labels_list=[os.path.join(path, img) for img in os.listdir(path) if img.endswith("_L.png")]
    img_path_features_list=[feature for feature in [os.path.join(path, img) for img in os.listdir(path)] if feature not in img_path_labels_list and feature.endswith(".png")] 
    return img_path_labels_list,img_path_features_list






if __name__ == "__main__":
    label,feature = read_data(path)
