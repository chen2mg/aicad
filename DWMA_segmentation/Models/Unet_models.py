# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 12:27:55 2019

@author: LIHAP9
"""
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, core, Dropout
from keras.models import Model

#Define the neural network. This will be used for training it from the scratch

##  ================================  ##

def get_unet(n_ch, patch_height, patch_width):
    
    input_patch = Input(shape=(n_ch, patch_height, patch_width))
 
    ##==========   left encoder of U-net =================##
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                   data_format='channels_first')(input_patch)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                   data_format='channels_first')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), data_format='channels_first')(conv1)
    
    
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                   data_format='channels_first')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                   data_format='channels_first')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2), data_format='channels_first')(conv2)
    
    
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                   data_format='channels_first')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                   data_format='channels_first')(conv3)                            ##===  same as VGG19  ====##
    pool3 = MaxPooling2D(pool_size=(2, 2), data_format='channels_first')(conv3)
      
    
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                   data_format='channels_first')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                   data_format='channels_first')(conv4)
    drop4 = Dropout(0.2)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2),data_format='channels_first')(drop4)

##==========   center of U-net =================##
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                   data_format='channels_first')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                   data_format='channels_first')(conv5)
    center = Dropout(0.2)(conv5)


##==========   right decoder of U-net =================##
    
    up6 = UpSampling2D(size=(2, 2), data_format='channels_first')(center)
    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal',
                 data_format='channels_first')(up6)
    merge6 = concatenate([drop4, up6], axis=1)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                   data_format='channels_first')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                   data_format='channels_first')(conv6)

    up7 = UpSampling2D(size=(2, 2), data_format='channels_first')(conv6)
    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal',
                 data_format='channels_first')(up7)
    merge7 = concatenate([conv3, up7], axis=1)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                   data_format='channels_first')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                   data_format='channels_first')(conv7)

    up8 = UpSampling2D(size=(2, 2), data_format='channels_first')(conv7)
    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal',
                 data_format='channels_first')(up8)
    merge8 = concatenate([conv2, up8], axis=1)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                   data_format='channels_first')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                   data_format='channels_first')(conv8)
    
    up9 = UpSampling2D(size=(2, 2), data_format='channels_first')(conv8)
    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal',
                 data_format='channels_first')(up9)
    merge9 = concatenate([conv1, up9], axis=1)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                   data_format='channels_first')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                   data_format='channels_first')(conv9)
    
##   shrink to 2 values  ##
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                   data_format='channels_first')(conv9)
   
##    use sigmoid output for Dice loss function   ##
    conv10 = Conv2D(1, 1, activation='sigmoid', data_format='channels_first')(conv9)

    model = Model(inputs=input_patch, outputs=conv10)

    model.summary()


    return model
