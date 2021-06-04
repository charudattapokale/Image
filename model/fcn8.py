# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 14:49:13 2021

@author: charu
"""

from keras.models import Model
from keras.layers import Input
from keras.layers.core import Lambda, Activation
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import Add
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras import backend as K

import tensorflow as tf


def dice_coef(y_true, y_pred):
    ''' metric dice coefficient'''
    
    return (2. * K.sum(y_true * y_pred) + 1.) / (K.sum(y_true) + K.sum(y_pred) + 1.)


def base(image_input):
    '''base architecture for fcn8 using vgg16'''
    
    #Block 1
        
    x = BatchNormalization()(image_input)
    x = Activation('relu')(x)

    x = Conv2D(64, (3, 3), padding='same', name='block1_conv2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = MaxPooling2D()(x)

    # Block 2
    x = Conv2D(128, (3, 3), padding='same', name='block2_conv1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(128, (3, 3), padding='same', name='block2_conv2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = MaxPooling2D()(x)

    # Block 3
    x = Conv2D(256, (3, 3), padding='same', name='block3_conv1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv2D(256, (3, 3), padding='same', name='block3_conv2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(256, (3, 3), padding='same', name='block3_conv3')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    block_3_out = MaxPooling2D()(x)

    # Block 4
    x = Conv2D(512, (3, 3), padding='same', name='block4_conv1')(block_3_out)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(512, (3, 3), padding='same', name='block4_conv2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(512, (3, 3), padding='same', name='block4_conv3')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    block_4_out = MaxPooling2D()(x)

    # Block 5
    x = Conv2D(512, (3, 3), padding='same', name='block5_conv1')(block_4_out)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(512, (3, 3), padding='same', name='block5_conv2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(512, (3, 3), padding='same', name='block5_conv3')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = MaxPooling2D()(x)
    
    return block_4_out, block_3_out,x


def top(intermediate_input,block_4_out,block_3_out,num_classes):
    '''top architecture of fcn8'''
    
    x = Conv2D(4096, (7, 7), activation='relu', padding='same')(intermediate_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(4096, (1, 1), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Classifying layers.
    x = Conv2D(num_classes, (1, 1), strides=(1, 1), activation='linear')(x)
    x = BatchNormalization()(x)

    block_3_out = Conv2D(num_classes, (1, 1), strides=(1, 1), activation='linear')(block_3_out)
    block_3_out = BatchNormalization()(block_3_out)

    block_4_out = Conv2D(num_classes, (1, 1), strides=(1, 1), activation='linear')(block_4_out)
    block_4_out = BatchNormalization()(block_4_out)

    x = Lambda(lambda x: tf.image.resize_images(x, (x.shape[1] * 2, x.shape[2] * 2)))(x)
    x = Add()([x, block_4_out])
    x = Activation('relu')(x)

    x = Lambda(lambda x: tf.image.resize_images(x, (x.shape[1] * 2, x.shape[2] * 2)))(x)
    x = Add()([x, block_3_out])
    x = Activation('relu')(x)

    x = Lambda(lambda x: tf.image.resize_images(x, (x.shape[1] * 8, x.shape[2] * 8)))(x)

    top_block_out = Activation('softmax')(x)
    
    return top_block_out 

        
def fcn8(num_classes, input_shape, lr_init, lr_decay, weight_path=None):
    '''complete fcn8 architecture combining base and top'''
    
    image_input = Input(input_shape)
    
    block_4_out,block_3_out,base_block_out = base(image_input)
    
    if weight_path is not None:
        base_model = Model(image_input, base_block_out)
        base_model.load_weights(weight_path, by_name=True)
        
    top_block_out = top(base_block_out,block_4_out,block_3_out,num_classes)
    
    model = Model(image_input, top_block_out )
    model.compile(optimizer=Adam(lr=lr_init, decay=lr_decay),
                  loss='categorical_crossentropy',
                  metrics=[dice_coef])

    return model
                        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    