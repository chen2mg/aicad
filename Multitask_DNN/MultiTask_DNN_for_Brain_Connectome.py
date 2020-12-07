
# -*- coding: utf-8 -*-


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from keras.utils import to_categorical
from DL_models import get_multitaskDNN_model

##  using synthetic data

input1_len = 4005
input2_len = 50

train_data1 = np.random.randn(80, input1_len)
train_data2 = np.random.randn(80, input2_len)
train_label1 = to_categorical(np.random.randn(80,)>0)
train_label2 = to_categorical(np.random.randn(80,)>0)
train_label3 = to_categorical(np.random.randn(80,)>0)
      
test_data1 = np.random.randn(80, input1_len)
test_data2 = np.random.randn(80, input2_len)
test_label1 = to_categorical(np.random.randn(80,)>0)
test_label2 = to_categorical(np.random.randn(80,)>0)
test_label3 = to_categorical(np.random.randn(80,)>0)

##  construct model 
model = get_multitaskDNN_model(input1_len, input2_len, finalAct="softmax")            

      
# Fit the model
model_training = model.fit([train_data1, train_data2], 
                           [train_label1, train_label2, train_label3],
                               epochs=10,
                               shuffle=True,
                               verbose=1,
                               batch_size=8)
                
##============  predict ============##
score = model.evaluate([test_data1, test_data2],
                       [test_label1, test_label2, test_label3])

print(model.metrics_names)
print(score) 
    




