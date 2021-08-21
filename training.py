from __future__ import print_function

import os
import matplotlib.pyplot as plt
import argparse
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping,ReduceLROnPlateau
from config.config import *
from network.fcn8 import fcn8
from dataloader.datareader import data_generator

print("**********Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
tf.debugging.set_log_device_placement(True)
# only 3 classes.
labels = ['background', 'person', 'car', 'road']


# model to train

model = fcn8(input_shape=INPUT_IMAGE_SHAPE, num_classes=len(labels),
                   lr_init=LEARNING_RATE, lr_decay=LR_DECAY, weight_path=WEIGHT_PATH)

# Define callbacks
checkpoint = ModelCheckpoint(filepath='./FCN8'+'_model_weight.h5',
                             monitor='val_dice_coef',
                             save_best_only=True,
                             save_weights_only=True)
#early_stopping = EarlyStopping(monitor='val_dice_coef', patience=5)
#reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,patience=5, min_lr=0.001)

# training
history = model.fit_generator(data_generator(TRAIN_DATASET_PATH, TRAIN_BATCH, 'train'),
                              steps_per_epoch= 81// TRAIN_BATCH,
                              validation_data=data_generator(TEST_DATASET_PATH, VAL_BATCH, 'val'),
                              validation_steps=20 // VAL_BATCH,
                              #callbacks=[checkpoint, train_check],
                              epochs=10,
                              verbose=1)

plt.title("loss")
plt.plot(history.history["loss"], color="r", label="train")
plt.plot(history.history["val_loss"], color="b", label="val")
plt.legend(loc="best")
plt.savefig(model_name + '_loss.png')

plt.gcf().clear()
plt.title("dice_coef")
plt.plot(history.history["dice_coef"], color="r", label="train")
plt.plot(history.history["val_dice_coef"], color="b", label="val")
plt.legend(loc="best")
plt.savefig(model_name + '_dice_coef.png')


    