# -*- coding: utf-8 -*-

from keras.models import Model
from keras.layers import Input, LeakyReLU, concatenate, add, Conv3D 
from keras.layers import BatchNormalization, MaxPooling3D, Conv3DTranspose, AveragePooling3D, ZeroPadding3D
from keras.optimizers import RMSprop, Adam, SGD
from keras import backend as K




def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)




def get_3D_unet_tiny(img_rows, img_cols, img_depth):
    
    N_base_filter = 8
    inputs = Input((img_rows, img_cols, img_depth, 1))
    
    conv1 = Conv3D(N_base_filter, (3, 3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv3D(N_base_filter, (3, 3, 3), activation='relu', padding='same')(conv1)
    conc1 = concatenate([inputs, conv1], axis=4)
    conc1 = BatchNormalization(axis=4)(conc1) 
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conc1)

    conv2 = Conv3D(N_base_filter*2, (3, 3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv3D(N_base_filter*2, (3, 3, 3), activation='relu', padding='same')(conv2)
    conc2 = concatenate([pool1, conv2], axis=4)
    conc2 = BatchNormalization(axis=4)(conc2) 

    up9 = concatenate([Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same')(conc2), conv1], axis=4)
    conv9 = Conv3D(N_base_filter, (3, 3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv3D(N_base_filter, (3, 3, 3), activation='relu', padding='same')(conv9)
    conc9 = concatenate([up9, conv9], axis=4)
    conc9 = BatchNormalization(axis=4)(conc9) 

    conv10 = Conv3D(1, (1, 1, 1), activation='sigmoid')(conc9)

    model = Model(inputs=[inputs], outputs=[conv10])

#    model.summary()
    for layer in model.layers:
        print(layer.output_shape)
    #plot_model(model, to_file='model.png')

#    model.compile(optimizer=RMSprop(lr=0.01, decay=1e-6),  
#                  loss= dice_coef_loss, 
#                  metrics= [dice_coef])
    
    model.compile(optimizer=Adam(lr=1e-2, beta_1=0.9, beta_2=0.999, 
                                 epsilon=1e-8, decay=1e-6),  
                  loss= dice_coef_loss, 
                  metrics= [dice_coef])

    return model




def get_3D_unet_small(img_rows, img_cols, img_depth):
    
    N_base_filter = 32
    inputs = Input((img_rows, img_cols, img_depth, 1))
    conv1 = Conv3D(N_base_filter, (3, 3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv3D(N_base_filter, (3, 3, 3), activation='relu', padding='same')(conv1)
    conc1 = concatenate([inputs, conv1], axis=4)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conc1)

    conv2 = Conv3D(N_base_filter*2, (3, 3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv3D(N_base_filter*2, (3, 3, 3), activation='relu', padding='same')(conv2)
    conc2 = concatenate([pool1, conv2], axis=4)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conc2)

    conv3 = Conv3D(N_base_filter*4, (3, 3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv3D(N_base_filter*4, (3, 3, 3), activation='relu', padding='same')(conv3)
    conc3 = concatenate([pool2, conv3], axis=4)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conc3)

    conv4 = Conv3D(N_base_filter*8, (3, 3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv3D(N_base_filter*8, (3, 3, 3), activation='relu', padding='same')(conv4)
    conc4 = concatenate([pool3, conv4], axis=4)

    up7 = concatenate([Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same')(conc4), conv3], axis=4)
    conv7 = Conv3D(N_base_filter*4, (3, 3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv3D(N_base_filter*4, (3, 3, 3), activation='relu', padding='same')(conv7)
    conc7 = concatenate([up7, conv7], axis=4)

    up8 = concatenate([Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(conc7), conv2], axis=4)
    conv8 = Conv3D(N_base_filter*2, (3, 3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv3D(N_base_filter*2, (3, 3, 3), activation='relu', padding='same')(conv8)
    conc8 = concatenate([up8, conv8], axis=4)

    up9 = concatenate([Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same')(conc8), conv1], axis=4)
    conv9 = Conv3D(N_base_filter, (3, 3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv3D(N_base_filter, (3, 3, 3), activation='relu', padding='same')(conv9)
    conc9 = concatenate([up9, conv9], axis=4)

    conv10 = Conv3D(1, (1, 1, 1), activation='sigmoid')(conc9)

    model = Model(inputs=[inputs], outputs=[conv10])

#    model.summary()
    for layer in model.layers:
        print(layer.output_shape)
    #plot_model(model, to_file='model.png')

    model.compile(optimizer=Adam(lr=1e-2, beta_1=0.9, beta_2=0.999, 
                                 epsilon=1e-8, decay=1e-6),  
                  loss= dice_coef_loss, 
                  metrics= [dice_coef])
        
#    model.compile(optimizer=RMSprop(lr=0.01, decay=1e-6), 
#              loss=dice_coef_loss, metrics=[dice_coef])
        

    return model



def get_3D_unet_large(img_rows, img_cols, img_depth):
    
    inputs = Input((img_rows, img_cols, img_depth, 1))
    conv1 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv1)
    conc1 = concatenate([inputs, conv1], axis=4)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conc1)

    conv2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv2)
    conc2 = concatenate([pool1, conv2], axis=4)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conc2)

    conv3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv3)
    conc3 = concatenate([pool2, conv3], axis=4)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conc3)

    conv4 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv4)
    conc4 = concatenate([pool3, conv4], axis=4)
    pool4 = MaxPooling3D(pool_size=(2, 2, 2))(conc4)

    conv5 = Conv3D(512, (3, 3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv3D(512, (3, 3, 3), activation='relu', padding='same')(conv5)
    conc5 = concatenate([pool4, conv5], axis=4)

    up6 = concatenate([Conv3DTranspose(256, (2, 2, 2), strides=(2, 2, 2), padding='same')(conc5), conv4], axis=4)
    conv6 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv6)
    conc6 = concatenate([up6, conv6], axis=4)

    up7 = concatenate([Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same')(conc6), conv3], axis=4)
    conv7 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv7)
    conc7 = concatenate([up7, conv7], axis=4)

    up8 = concatenate([Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(conc7), conv2], axis=4)
    conv8 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv8)
    conc8 = concatenate([up8, conv8], axis=4)

    up9 = concatenate([Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same')(conc8), conv1], axis=4)
    conv9 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv9)
    conc9 = concatenate([up9, conv9], axis=4)

    conv10 = Conv3D(1, (1, 1, 1), activation='sigmoid')(conc9)

    model = Model(inputs=[inputs], outputs=[conv10])

    model.summary()
    for layer in model.layers:
        print(layer.output_shape)
    #plot_model(model, to_file='model.png')

    model.compile(optimizer=Adam(lr=1e-2, beta_1=0.9, beta_2=0.999, 
                                 epsilon=1e-8, decay=1e-6),  
                  loss= dice_coef_loss, 
                  metrics= [dice_coef])

    return model