# -*- coding: utf-8 -*-
"""
@author: Hailong Li
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#  Imports
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.optimizers import  Adam
from keras.models import Model
from keras.applications.vgg19 import VGG19
from keras.applications.inception_resnet_v2 import preprocess_input
from DL_models import DeepMultimodal



##  create random samples
FC_features = np.random.rand(72,90,90,3)
SC_features = np.random.rand(72,90,90,3)
DWMA_features = np.random.rand(72,10)
Clinical_features = np.random.rand(72,72)
labels = np.random.rand(72)>0.5
sub_idx = np.array((range(72)))

## separate training and testing
train_idx, test_idx = train_test_split(sub_idx, shuffle=False)


FC_train = FC_features[train_idx,:,:,:]
SC_train = SC_features[train_idx,:,:,:]
DWMA_train = DWMA_features[train_idx,:]
Clinical_train = Clinical_features[train_idx,:]
labels_train = to_categorical(labels[train_idx])


FC_test = FC_features[test_idx,:,:,:]
SC_test = SC_features[test_idx,:,:,:]
DWMA_test = DWMA_features[test_idx,:]
Clinical_test = Clinical_features[test_idx,:]
labels_test = to_categorical(labels[test_idx])




##  load different pretrained network
base_model = VGG19(include_top=False, weights='imagenet', input_shape=(90,90,3))
TL_model = Model(inputs=base_model.input, outputs=base_model.output)
for layer in TL_model.layers:
      layer.trainable=False
      
    ##   TL with VGG
         #  TL for train 
FC_train = preprocess_input(FC_train)
SC_train = preprocess_input(SC_train)

      
FC_train = TL_model.predict(FC_train)
SC_train = TL_model.predict(SC_train)

         #  TL for test 
FC_test = preprocess_input(FC_test)
SC_test = preprocess_input(SC_test)

        
FC_test = TL_model.predict(FC_test)
SC_test = TL_model.predict(SC_test)
  
    

    
#==============================================================================
##============  construct CNN based on keras ============##
FC_size = FC_train.shape[2]
SC_size = SC_train.shape[2]
chanel_size = FC_train.shape[3]
dwma_len = DWMA_train.shape[1]
clin_len = Clinical_train.shape[1]

model = DeepMultimodal.build(FC_size, chanel_size, SC_size, dwma_len, clin_len) 
#                        16, third_kern_size, 32, four_kern_size)

# Compile the model
lr = 0.01
epoch_max = 10
my_optimizer = Adam(lr=lr, decay=lr/epoch_max)
model.compile(optimizer=my_optimizer,loss='categorical_crossentropy',metrics=['accuracy'])         


# Fit the model/ generator
model_training = model.fit([FC_train, SC_train, DWMA_train, Clinical_train],
                            labels_train, 
                           epochs=epoch_max,
                           shuffle=False,
                           verbose=1,
                           batch_size=8)


#  predict    
score = model.evaluate([FC_test, SC_test, DWMA_test, Clinical_test],labels_test)
print('Test accuracy is {}'.format(score[1]))    
