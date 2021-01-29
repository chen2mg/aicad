# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 13:00:51 2018
learn  CNN
@author: LIHAP9
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#  Imports
import numpy as np
from sklearn.model_selection import train_test_split
import keras
from keras import layers

##  load data 

#SEEG_df = pd.read_excel('Data\data.xlsx', header=None)

##   format data
target_features =  np.random.rand(50, 4005)  
labels = np.zeros(50)
labels[:25,] = 1

source_features = np.random.rand(800, 4005)


##   step 1: build SSAE model, and train with source features  ##
input_features = keras.Input(shape=(4005,))
encoded = layers.Dense(128, activation='relu')(input_features)
encoded = layers.Dense(64, activation='relu')(encoded)

decoded = layers.Dense(128, activation='relu')(encoded)
decoded = layers.Dense(4005, activation='sigmoid')(decoded)

##   
autoencoder = keras.Model(input_features, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.fit(source_features, source_features,
                epochs=5,
                batch_size=64,
                shuffle=True)


##  step 2:  re-use encoder in a DNN model, and train with target features ##

X_train, X_test, y_train, y_test = train_test_split(target_features, labels, 
                                                    test_size=0.33, random_state=42)

y_train = keras.utils.to_categorical(y_train) 
y_test = keras.utils.to_categorical(y_test) 

encoded_features = layers.Dense(16, activation='relu')(encoded)
encoded_features = layers.Dense(8, activation='relu')(encoded_features)
output = layers.Dense(2, activation='softmax')(encoded_features)

DNN_Model = keras.Model(input_features, output)
DNN_Model.compile(optimizer='adam', loss='binary_crossentropy')
DNN_Model.fit(X_train, y_train,
                epochs=200,
                batch_size=8,
                shuffle=True)


## ==== test model ========##
    
predictions = DNN_Model.predict(X_test)
model_acc = DNN_Model.evaluate(X_test, y_test)

print('Test accuracy is {}'.format(model_acc))   
     
w1 = autoencoder.get_weights()
w2 = DNN_Model.get_weights()

  