# -*- coding: utf-8 -*-


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from keras.utils import to_categorical
from DL_models import get_TLCNN_model




##  using synthetic data
input_len = 90
train_data = np.random.randn(80, input_len, input_len, 3)
train_label = to_categorical(np.random.randn(80,)>0)
    
test_data = np.random.randn(80, input_len, input_len, 3)
test_label = to_categorical(np.random.randn(80,)>0)

##  construct model 
model = get_TLCNN_model(input_len) 
      
# Fit the model
model_training = model.fit(train_data, train_label,
                               epochs=10,
                               shuffle=True,
                               verbose=1,
                               batch_size=8)
                
##============  predict  ============##
score = model.evaluate(train_data,train_label)
print('Validation accuracy is {}'.format(score[1])) 
    
