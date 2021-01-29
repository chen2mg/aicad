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
import pandas as pd
import scipy.io as sio
#from os import listdir
from sklearn import preprocessing
from keras.utils import to_categorical
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import keras
from keras import layers

##  load data 

#SEEG_df = pd.read_excel('Data\data.xlsx', header=None)

##   format data
target_features =  np.random.rand(28, 4005)  
labels = np.zeros(28)
labels[:14,] = 1

source_features = np.random.rand(800, 4005)
##   set-up experiments

##   build SSAE model  ##
input_features = keras.Input(shape=(4005,))
encoded = layers.Dense(128, activation='relu')(input_features)
encoded = layers.Dense(64, activation='relu')(encoded)

decoded = layers.Dense(128, activation='relu')(encoded)
decoded = layers.Dense(4005, activation='sigmoid')(decoded)

autoencoder = keras.Model(input_features, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.fit(source_features, source_features,
                epochs=20,
                batch_size=64,
                shuffle=True)


encoder = keras.Model(input_features, encoded)
##  data split        

X_train, X_test, y_train, y_test = train_test_split(target_features, labels, 
                                                    test_size=0.33, random_state=42)

##  use encoder part on target domain to reduce feature dimension
encoded_features_train = encoder.predict(X_train)
encoded_features_test = encoder.predict(X_test)
    
##============  SVM model ============##
svmModel = SVC(kernel='linear', C=1.0)
svmModel.fit(encoded_features_train, y_train) 
   
## ==== test model ========##
    
predictions = svmModel.predict(encoded_features_test)
model_acc = svmModel.score(encoded_features_test, y_test)

print('Test accuracy is {}'.format(model_acc))   
     
  