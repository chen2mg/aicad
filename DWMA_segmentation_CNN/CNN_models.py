# -*- coding: utf-8 -*-

from keras.models import Model
from keras.layers import Sequenctial, Conv2D 
from keras.layers import MaxPooling2D, 
from keras.optimizers import RMSprop, Adam, SGD
from keras import backend as K





def get_CNN(ksize, input_shape):
    
##  start experimenting networks
    model = Sequential()
    ##   first Cov layer
    model.add(Conv2D(8, kernel_size=(ksize, ksize), strides=(1, 1),
                     activation=act_fun,
                     padding="valid",
                     kernel_initializer=initializers.glorot_uniform(),
                     input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    #model.add(Dropout(0.4))
    
    
    ## second Conv layer
    model.add(Conv2D(8, kernel_size=(ksize, ksize), 
                     activation=act_fun,
                     kernel_initializer=initializers.glorot_uniform(),
                     padding="valid"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    #model.add(Dropout(0.4)
    
    ## third Conv layer
    model.add(Conv2D(16, (ksize, ksize), 
           activation='relu',
           kernel_initializer=initializers.glorot_uniform(),
           padding="valid"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    #model.add(Dropout(0.4))
    
    ##  fully connected layers 1
    model.add(Flatten())
    model.add(Dense(10,kernel_initializer=initializers.glorot_uniform()))
    model.add(BatchNormalization())
    model.add(Activation(act_fun))
    #model.add(Dropout(0.4))
    
    ##  fully connected layers 2
#    model.add(Dense(20,kernel_initializer=initializers.glorot_uniform()))
#    model.add(BatchNormalization())
#    model.add(Activation(act_fun))
    #model.add(Dropout(0.4))
    
    # output softmax layer
    model.add(Dense(2,kernel_initializer=initializers.glorot_uniform()))
    model.add(BatchNormalization())
    model.add(Activation('softmax'))
    
    # Compile the model
    sgd = SGD(lr=0.1,momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd,loss='categorical_crossentropy',metrics=['accuracy']) 

    return model