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


##  load data 

#SEEG_df = pd.read_excel('Data\data.xlsx', header=None)

##   format data
features =  np.random.rand(100,64)  # assume 64 feature x 100 samples
labels = np.zeros(100)
labels[:50,] = 1
##   set-up experiments


Fold = 10
i=1
skf = StratifiedKFold(n_splits=Fold, random_state=42, shuffle=False)

All_true_list= 100*np.ones(labels.shape)
All_prediction_list = 100*np.ones(labels.shape)

for idx1, idx2 in skf.split(features, labels):
       
    print('\nTesting model in fold : %i\n'%(i))
    i = i+1
    train_data=features[idx1,:]
    train_label=labels[idx1]
    
    test_data = features[idx2,:]
    test_label = labels[idx2]
    
    #  normalization
    scaler = preprocessing.StandardScaler().fit(train_data)
    train_data = scaler.transform(train_data)
    min_max_scaler = preprocessing.MinMaxScaler()
    train_data = min_max_scaler.fit_transform(train_data)
    train_label =train_label.astype(int)
    
    test_data = scaler.transform(test_data)
    test_data = min_max_scaler.transform(test_data)
    test_label =test_label.astype(int)
        
    #==============================================================================
    ##============  SVM model ============##
    Model = SVC(kernel='linear', C=1.0)
    Model.fit(train_data, train_label) 

    #==============================================================================
    ##============  LR model ============##
#    Model = LogisticRegression(solver='lbfgs')
#    Model.fit(train_data, train_label) 
    
    #==============================================================================
    ##============  rbf SVM model ============##
#    Model = SVC(kernel='rbf', C=100.0, gamma='auto')
#    Model.fit(train_data, train_label) 
    
    #==============================================================================
    ##============  k-NN model ============##   
#    Num_Neighbor = 5
#    Model = KNeighborsClassifier(n_neighbors=Num_Neighbor)
#    Model.fit(train_data, train_label) 
    
    
    
    ## ==== test model ========##
    predictions = Model.predict(test_data)
    model_acc = Model.score(test_data,test_label)
    print('Test accuracy is {}'.format(model_acc))   
     

    All_prediction_list[idx2,] = predictions
    All_true_list[idx2,] = test_label
    