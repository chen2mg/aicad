
# -*- coding: utf-8 -*-


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from keras.utils import to_categorical
from DL_models import get_multiscaleDNN_model

##  using synthetic data

input1_len = int(90*(90-1)*0.5)
input2_len = int(200*(200-1)*0.5)
input3_len = int(400*(400-1)*0.5)
input4_len = 5

train_data1 = np.random.randn(80, input1_len)
train_data2 = np.random.randn(80, input2_len)
train_data3 = np.random.randn(80, input3_len)
train_data4 = np.random.randn(80, input4_len)
train_label = to_categorical(np.random.randn(80,)>0)
    
test_data1 = np.random.randn(80, input1_len)
test_data2 = np.random.randn(80, input2_len)
test_data3 = np.random.randn(80, input3_len)
test_data4 = np.random.randn(80, input4_len)
test_label = to_categorical(np.random.randn(80,)>0)


##  construct model 
model = get_multiscaleDNN_model(input1_len, input2_len, input3_len, input4_len)            

      
# Fit the model
model_training = model.fit([train_data1, train_data2, train_data3, train_data4], train_label,
                               epochs=10,
                               shuffle=True,
                               verbose=1,
                               batch_size=8)
                
##============  predict ============##
score = model.evaluate([test_data1, test_data2, test_data3, test_data4],train_label)
print('Validation accuracy is {}'.format(score[1])) 
    
