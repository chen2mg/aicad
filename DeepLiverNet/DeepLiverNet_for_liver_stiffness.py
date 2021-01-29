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
##============  fix randomness and reproduce results ============##
import os
import random as rn
import tensorflow as tf
rn.seed(12345)
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

from keras import backend as K
tf.set_random_seed(1234)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

####################################
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn import preprocessing

from keras.utils import to_categorical
from keras.models import load_model
from keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

from sklearn.model_selection import KFold
from my_models.CNN_model import CNN
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

from DL_models import DeepLiverNet
from keras.models import Model
from keras.applications.vgg19 import VGG19
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.nasnet import NASNetLarge
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_resnet_v2 import preprocess_input
K.clear_session()    #   remove model from memory



##  load data

#List_mat_content = sio.loadmat('Data\\LS_deep_data.mat')
List_mat_content = sio.loadmat('...your Data file...')
Mean_score = List_mat_content['MRE_outcomes']
Clinical_features = List_mat_content['Clinical_features']
Radiomics_features = List_mat_content['Radiomics_features']
Scanner_features = List_mat_content['Scanner_features']
Clinical_features  = np.concatenate([Clinical_features, Scanner_features], axis=1)



Path = 'Data\\Data_for_VGG\\'
LS_cut = 3.0
Fold = 10
epoch_max = 3000
i=1


LS_labels = Mean_score >= LS_cut
kf = KFold(n_splits=Fold, random_state=42, shuffle=False)


All_true_list=[]
All_score_list = []
All_prediction_list=[]

##  load different pretrained network
base_model = VGG19(include_top=False, weights='imagenet', input_shape=(224,224,3))
#base_model = InceptionResNetV2(include_top=False, weights='imagenet', input_shape=(224,224,3))
#base_model = NASNetLarge(include_top=False, weights='imagenet', input_shape=(224,224,3))


TL_model = Model(inputs=base_model.input, outputs=base_model.output)
for layer in TL_model.layers:
      layer.trainable=False


for idx1, idx2 in kf.split(your_train_data, your_labels):
    
    
    print('\nTraining and testing model in fold : %i\n'%(i))
    i = i+1
    train_data_1=[]
    train_data_2=[]
    train_data_3=[]
    train_data_4=[]
    train_data_c=[]
    
    train_label=[]
    
    test_data_1 = []
    test_data_2 = []
    test_data_3 = []
    test_data_4 = []
    test_data_c = []
    
    test_label = []
    test_score = []       
        
#    test_data=test_data.reshape(test_data.shape[0], size_img*size_img)
    test_data_c=np.vstack(test_data_c)
    test_label=np.vstack(test_label)
    test_score=np.vstack(test_score) 
    
    train_label = to_categorical(train_label.astype(int))
    test_label = to_categorical(test_label.astype(int))
    
    #  norm clinical data    
    scaler_c = preprocessing.StandardScaler().fit(train_data_c)
    train_data_c = scaler_c.transform(train_data_c)            
    test_data_c = scaler_c.transform(test_data_c)
       
    ##   TL with VGG
         #  TL for train 
    train_data_1 = preprocess_input(train_data_1)
    train_data_2 = preprocess_input(train_data_2)
    train_data_3 = preprocess_input(train_data_3)
    train_data_4 = preprocess_input(train_data_4)
        
    train_data_1 = TL_model.predict(train_data_1)
    train_data_2 = TL_model.predict(train_data_2)
    train_data_3 = TL_model.predict(train_data_3)
    train_data_4 = TL_model.predict(train_data_4)
         #  TL for test 
    test_data_1 = preprocess_input(test_data_1)
    test_data_2 = preprocess_input(test_data_2)
    test_data_3 = preprocess_input(test_data_3)
    test_data_4 = preprocess_input(test_data_4)
        
    test_data_1 = TL_model.predict(test_data_1)
    test_data_2 = TL_model.predict(test_data_2)
    test_data_3 = TL_model.predict(test_data_3)
    test_data_4 = TL_model.predict(test_data_4)    
    
    #     weighted loss
    y_integers = np.argmax(train_label, axis=1)
    class_weights = compute_class_weight('balanced', np.unique(y_integers), y_integers)
    LS_class_weights = dict(enumerate(class_weights))
    
    #==============================================================================
    ##============  construct CNN based on keras ============##
    img_size = train_data_1.shape[2]
    chanl_size = train_data_1.shape[3]
    clin_len = train_data_c.shape[1]
    
    model = DeepLiverNet.build(img_size, chanl_size, clin_len) 
#                        16, third_kern_size, 32, four_kern_size)

    # Compile the model
    lr = 0.01
    my_optimizer = Adam(lr=lr, decay=lr/epoch_max)
    model.compile(optimizer=my_optimizer,loss='categorical_crossentropy',metrics=['accuracy'])         
    
    #   early stop and checkpoint
    es = EarlyStopping(monitor='val_acc', mode='max', verbose=1, patience=epoch_max)
    mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)
    
    # Fit the model/ generator
    model_training = model.fit([train_data_1, train_data_2, train_data_3, train_data_4, train_data_c],
                                train_label, 
                               epochs=epoch_max,
                               shuffle=False,
                               verbose=1,
                               batch_size=8,
                               class_weight = LS_class_weights,
                               callbacks=[es, mc])
    
    
    ##============  predict using CNN from keras ============##
        ##     load test data with Normalization
    saved_model = load_model('best_model.h5')
    predictions = saved_model.predict([test_data_1, test_data_2, test_data_3, test_data_4, test_data_c])
    score = saved_model.evaluate([test_data_1, test_data_2, test_data_3, test_data_4, test_data_c],test_label)
    print('Test accuracy is {}'.format(score[1]))     

    All_prediction_list.append(predictions)
    All_true_list.append(test_label)
    All_score_list.append(test_score)
