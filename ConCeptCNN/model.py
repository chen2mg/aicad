# Ming Chen
import numpy as np
import scipy.io as sio
import os
import math
# plot the model
os.environ["PATH"] += os.pathsep + 'C:\\Program Files (x86)\\Graphviz2.38\\bin'
from keras.utils.vis_utils import plot_model

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, accuracy_score
from keras.layers import Dense, Dropout, BatchNormalization, Activation, Input, Conv2D, Flatten, MaxPooling2D, GlobalAveragePooling2D, AveragePooling2D, concatenate
from keras.models import Sequential, Model
import keras
from indeption_model import inception_module
import scipy.io as sio
import jason
import matplotlib.pyplot as plt

def check_models():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 1), padding='valid', activation='relu', input_shape=(90, 90, 1), name='conv1'))
    model.add(AveragePooling2D((3, 1), strides=(2, 1), name='AVG_pool1'))
    model.add(Conv2D(64, kernel_size=(3, 1), padding='valid', activation='relu', name='conv2'))
    model.add(AveragePooling2D((3, 1), strides=(2, 1), name='AVG_pool2'))
    model.summary()

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5, 1), padding='valid', activation='relu', input_shape=(90, 90, 1), name='conv1'))
    model.add(AveragePooling2D((3, 1), strides=(2, 1), name='AVG_pool1'))
    model.add(Conv2D(64, kernel_size=(5, 1), padding='valid', activation='relu', name='conv2'))
    model.add(AveragePooling2D((3, 1), strides=(2, 1), name='AVG_pool2'))
    model.summary()

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(7, 1), padding='valid', activation='relu', input_shape=(90, 90, 1), name='conv1'))
    model.add(AveragePooling2D((3, 1), strides=(2, 1), name='AVG_pool1'))
    model.add(Conv2D(64, kernel_size=(7, 1), padding='valid', activation='relu', name='conv2'))
    model.add(AveragePooling2D((3, 1), strides=(2, 1), name='AVG_pool2'))
    model.summary()

    model = Sequential()
    model.add(Conv2D(64, kernel_size=(90, 1), padding='valid', activation='relu', input_shape=(90, 90, 1), name='conv1'))
    model.summary()
    
    
def build_fc(input_shape=(90, 90)):
    input_data = Input(shape=input_shape)

    coarse_channel = Conv2D(64, kernel_size=(1, 3), padding='valid', activation='relu', name='coarse_conv1')(input_data)
    coarse_channel = AveragePooling2D((1, 3), strides=(1, 2), name='coarse_AVG_pool1')(coarse_channel)
    coarse_channel = Conv2D(128, kernel_size=(1, 3), padding='valid', activation='relu', name='coarse_conv2')(coarse_channel)
    coarse_channel = AveragePooling2D((1, 3), strides=(1, 2), name='coarse_AVG_pool2')(coarse_channel)

    medium_channel = Conv2D(64, kernel_size=(1, 5), padding='valid', activation='relu', name='medium_conv1')(input_data)
    medium_channel = AveragePooling2D((1, 3), strides=(1, 2), name='medium_AVG_pool1')(medium_channel)
    medium_channel = Conv2D(128, kernel_size=(1, 5), padding='valid', activation='relu', name='medium_conv2')(medium_channel)
    medium_channel = AveragePooling2D((1, 3), strides=(1, 2), name='medium_AVG_pool2')(medium_channel)

    fine_channel = Conv2D(64, kernel_size=(1, 7), padding='valid', activation='relu', name='fine_conv1')(input_data)
    fine_channel = AveragePooling2D((1, 3), strides=(1, 2), name='fine_AVG_pool1')(fine_channel)
    fine_channel = Conv2D(128, kernel_size=(1, 7), padding='valid', activation='relu', name='fine_conv2')(fine_channel)
    fine_channel = AveragePooling2D((1, 3), strides=(1, 2), name='fine_AVG_pool2')(fine_channel)

    global_channel = Conv2D(128, kernel_size=(1, 90), padding='valid', activation='relu', name='global_conv1')(input_data)

    # merge filted data
    img_feat = concatenate([coarse_channel, medium_channel, fine_channel, global_channel], axis=2)
    img_feat = Flatten()(img_feat)

    img_feat = Dense(256, use_bias=False, name='dense1')(img_feat)
    img_feat = Dropout(0.5)(img_feat)
    img_feat = BatchNormalization()(img_feat)
    img_feat = Dense(256, use_bias=False, name='dense2')(img_feat)
    img_feat = Dropout(0.5)(img_feat)
    img_feat = BatchNormalization()(img_feat)
    out = Dense(1, use_bias=False)(img_feat)
    out = Activation('sigmoid', name='prediction_layer')(out)
    model = Model(inputs=input_data,
                    outputs=out,
                    name="Multi-filter-CNN")
    return model
    
