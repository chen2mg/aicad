# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 10:55:53 2018

@author: LIHAP9
"""

from keras.layers import Dense, BatchNormalization, Activation, Input, concatenate
#from keras.layers import Concatenate
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, GlobalAveragePooling2D
import os
import random as rn
import tensorflow as tf
rn.seed(12345)
os.environ['PYTHONHASHSEED'] = '0'

            
##################################################################################
          
class DeepLiverNet:
    """
    ======================= Branch =================================
    """
#    @staticmethod
    def build_CNN_input(inputs, name="Branch_input"):  

        ##   first Cov layer
        size_pool = 1
        size_stride = 1
        firstL = 8
        firstK = 3
        secondL = 16
        secondK = 3
        branch_input = Conv2D(firstL, kernel_size=(firstK, firstK),
                         strides=(size_stride, size_stride),
                         padding="valid")(inputs)         
        branch_input =BatchNormalization()(branch_input)
        branch_input =Activation('relu')(branch_input)      
        branch_input =MaxPooling2D(pool_size=(size_pool, size_pool),
                                   strides=(size_stride, size_stride))(branch_input)
        
        ## second Conv layer
        branch_input = Conv2D(secondL, kernel_size=(secondK, secondK),
                         strides=(size_stride, size_stride),
                         padding="valid")(branch_input)  
        branch_input =BatchNormalization()(branch_input)
        branch_input =Activation('relu')(branch_input)
        branch_input =MaxPooling2D(pool_size=(size_pool, size_pool),
                                   strides=(size_stride, size_stride))(branch_input)
#        branch_input = Dropout(0.2)(branch_input) 

        
        ##   from Conv to FC
#        branch_input = inputs
#        branch_input = GlobalAveragePooling2D()(branch_input)
        branch_input = Flatten()(branch_input)
        
        ##  fully connected layers 1
        branch_input = Dense(8)(branch_input)
        branch_input = BatchNormalization()(branch_input)
        branch_input = Activation('relu', name = name)(branch_input)  

        
        return branch_input
              
    """
    =======================multi-input deep CNN model =================================
    """    
#    @staticmethod
    def build(size_img, size_chanl, clin_len):
        
        input1 = Input(shape=(size_img,size_img,size_chanl,))   #   
        input2 = Input(shape=(size_img,size_img,size_chanl,))   #
        input3 = Input(shape=(size_img,size_img,size_chanl,))
        input4 = Input(shape=(size_img,size_img,size_chanl,))
        inputc = Input(shape=(clin_len,))
        
        ##   Img input feature extraction       
        input1_feat = DeepLiverNet.build_CNN_input(input1, name="Branch_input1")
        input2_feat = DeepLiverNet.build_CNN_input(input2, name="Branch_input2")
        input3_feat = DeepLiverNet.build_CNN_input(input3, name="Branch_input3")
        input4_feat = DeepLiverNet.build_CNN_input(input4, name="Branch_input4")
        
        #    merge img data
        img_feat = concatenate([input1_feat, input2_feat, input3_feat,
                               input4_feat], axis=1)
        
        #    layer after merge
        img_feat = Dense(8)(img_feat)
        img_feat = BatchNormalization()(img_feat)
        img_feat = Activation('relu')(img_feat)
#        img_feat = Dropout(0.2)(img_feat)
        
        ##########
        #   input clinical feature extraction 
        ##########
        clin_feat = Dense(8)(inputc)
        clin_feat = BatchNormalization()(clin_feat)
        clin_feat = Activation('relu')(clin_feat)
     
        ##############
        #   merge high level integrated features
        ##############
        hl_feat = concatenate([img_feat, clin_feat], axis=1)
        
         #   second layer after merge
        hl_feat = Dense(16)(hl_feat)
        hl_feat = BatchNormalization()(hl_feat)
        hl_feat = Activation('relu')(hl_feat) 
        hl_feat = Dropout(0.2)(hl_feat)
        
        hl_feat = Dense(8)(hl_feat)
        hl_feat = BatchNormalization()(hl_feat)
        hl_feat = Activation('relu')(hl_feat) 
        
        ##   output
        hl_feat = Dense(2)(hl_feat)
        hl_feat = BatchNormalization()(hl_feat)
        Output = Activation('softmax')(hl_feat)       

        
        ##   combine
        model = Model(inputs=[input1, input2, input3, input4, inputc], 
                      outputs = Output,
                      name = "LiverCNN")
        
        return model
        