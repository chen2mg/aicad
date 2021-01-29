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
import os
import random as rn
from sklearn import preprocessing
from CNN_models import get_CNN


patch_s = 13
epoch_max = 5000
start_fold_idx = 0
sample_size  =50
size_img = 13
act_fun = 'relu'


##============  load data ============##
train_data = ...
train_labels = ...

##  normalization
scaler = preprocessing.StandardScaler().fit(train_data)
train_data = scaler.transform(train_data)
min_max_scaler = preprocessing.MinMaxScaler()
train_data = min_max_scaler.fit_transform(train_data)
train_data=train_data.reshape(train_data.shape[0], size_img, size_img, 1)

    ##============  construct CNN from keras ============##
    
input_shape = (size_img,size_img,1)
ksize = 3
model = get_CNN( ksize , input_shape) 
# Fit the model
model_training = model.fit(train_data,train_labels,
                       validation_split=0.2, 
                           epochs=epoch_max,
                           shuffle=True,
                           verbose=1,
                           batch_size=256)
    

    
    ##============  save DNN model from keras ============##
model.save('CNN_1.h5')
   
    
    ##============  predict using CNN from keras ============##
        ##     load test data with Normalization 
predictions = model.predict(test_data)
score = model.evaluate(test_data,test_labels)
#print('Test accuracy is {}'.format(score[1]))   
#probability_true = predictions[:,1]
#print(predictions[:10,])        
seg = predictions[:,0]<predictions[:,1]
gt = test_labels[:,0]<test_labels[:,1]
dice = np.sum(seg[gt==1])*2.0 / (np.sum(seg) + np.sum(gt))
        
 