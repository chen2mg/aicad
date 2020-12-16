
from keras.optimizers import SGD
from keras.applications.vgg19 import VGG19
from keras.layers import Dense, Dropout, BatchNormalization, Activation, concatenate
from keras.models import Sequential, Model
from keras.layers import Flatten, Conv2D, Input


def get_TLCNN_model(input_len):
        TL_model = VGG19(include_top=False, weights='imagenet', input_shape=(input_len,input_len,3))
        for layer in TL_model.layers:
                  layer.trainable=False
                                  
        model = Sequential()
        model.add(TL_model)
        
        model.add(Conv2D(32, kernel_size=(1, 1),activation='relu'))
        model.add(BatchNormalization())
        
        model.add(Conv2D(64, kernel_size=(1, 1),activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))          
                
        model.add(Flatten())
        model.add(Dense(10))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        
        model.add(Dense(2))
        model.add(BatchNormalization())
        model.add(Activation('softmax'))                 
                
        # Compile the model
        sgd = SGD(lr=0.01,momentum=0.9, nesterov=True)
        model.compile(optimizer=sgd,loss='categorical_crossentropy',metrics=['accuracy']) 
        # model.summary()
        return model


def get_multiscaleDNN_model(input1_len, input2_len, input3_len, input4_len):
        input1 = Input(shape=(input1_len,))
        input2 = Input(shape=(input2_len,))
        input3 = Input(shape=(input3_len,))
        input4 = Input(shape=(input4_len,))
        #  Branch 1
        branch1 = Dense(2048, use_bias=False)(input1)
        branch1 = BatchNormalization()(branch1)
        branch1 = Activation('relu')(branch1)
        branch1 = Dropout(0.2)(branch1)
        #
        branch1 = Dense(512)(branch1)
        branch1 = BatchNormalization()(branch1)
        branch1 = Activation('relu')(branch1)
        branch1 = Dropout(0.2)(branch1)

        branch1 = Dense(8)(branch1)
        branch1 = BatchNormalization()(branch1)
        branch1 = Activation('relu')(branch1)
        branch1 = Dropout(0.2)(branch1)

        #   Branch 2
        branch2 = Dense(2048, use_bias=False)(input2)
        branch2 = BatchNormalization()(branch2)
        branch2 = Activation('relu')(branch2)
        branch2 = Dropout(0.2)(branch2)
        #
        branch2 = Dense(256)(branch2)
        branch2 = BatchNormalization()(branch2)
        branch2 = Activation('relu')(branch2)
        branch2 = Dropout(0.2)(branch2)

        branch2 = Dense(8)(branch2)
        branch2 = BatchNormalization()(branch2)
        branch2 = Activation('relu')(branch2)
        branch2 = Dropout(0.2)(branch2)

        #   Branch 3
        branch3 = Dense(512, use_bias=False)(input3)
        branch3 = BatchNormalization()(branch3)
        branch3 = Activation('relu')(branch3)
        branch3 = Dropout(0.2)(branch3)
        #
        branch3 = Dense(64)(branch3)
        branch3 = BatchNormalization()(branch3)
        branch3 = Activation('relu')(branch3)
        branch3 = Dropout(0.2)(branch3)

        branch3 = Dense(8)(branch3)
        branch3 = BatchNormalization()(branch3)
        branch3 = Activation('relu')(branch3)
        branch3 = Dropout(0.2)(branch3)

        #  Branch 4
        branch4 = Dense(16, use_bias=False)(input4)
        branch4 = BatchNormalization()(branch4)
        branch4 = Activation('relu')(branch4)
        branch4 = Dropout(0.2)(branch4)

        #   merge
        concat_feat = concatenate([branch1, branch2, branch3, branch4], axis=1)

        # output layer
        concat_feat = Dense(10, use_bias=False)(concat_feat)
        concat_feat = BatchNormalization()(concat_feat)
        concat_feat = Activation('relu')(concat_feat)
        concat_feat = Dropout(0.1)(concat_feat)

        concat_feat = Dense(2, use_bias=False)(concat_feat)
        concat_feat = BatchNormalization()(concat_feat)
        concat_feat = Activation('softmax')(concat_feat)
        outputs = Dropout(0.1)(concat_feat)

        # final model
        model = Model(inputs=[input1, input2, input3, input4],
                      outputs=outputs,
                      name="multi_input_DNN")
        sgd = SGD(lr=0.01,momentum=0.9, nesterov=True)
        model.compile(optimizer=sgd,loss='categorical_crossentropy',metrics=['accuracy']) 
        # model.summary()
        return model



##   MIMT
def build_branch(inputs,finalAct="softmax", name="Branch_output"):      
        branch = Dense(2,use_bias=False)(inputs)
        branch = BatchNormalization()(branch)
        branch = Activation(finalAct, name = name)(branch)      
        return branch    
  
    
def get_multitaskDNN_model(input1_len, input2_len, finalAct="softmax"):
        input1 = Input(shape=(input1_len,))   #   connectome map
        input2 = Input(shape=(input2_len,))   #    clinical

        #   input 1 feature extraction       
        input1_feat = Dense(128, use_bias=False)(input1)
        input1_feat = BatchNormalization()(input1_feat)
        input1_feat = Activation('relu')(input1_feat)


        input1_feat = Dense(32, use_bias=False, )(input1_feat)
        input1_feat = BatchNormalization()(input1_feat)
        input1_feat = Activation('relu')(input1_feat)

        #   input 2 feature extraction       
        input2_feat = Dense(16, use_bias=False)(input2)
        input2_feat = BatchNormalization()(input2_feat)
        input2_feat = Activation('relu')(input2_feat)
        
        #   merge
        hl_feat = concatenate([input1_feat, input2_feat], axis=1)
        
        #   second layer after merge
        hl_feat = Dense(8, use_bias=False)(hl_feat)
        hl_feat = BatchNormalization()(hl_feat)
        hl_feat = Activation('relu')(hl_feat)
        
        #   task_Num  branch       
        Branch_1 = build_branch(hl_feat, finalAct=finalAct, name="Branch1_output")
        Branch_2 = build_branch(hl_feat, finalAct=finalAct, name="Branch2_output")
        Branch_3 = build_branch(hl_feat, finalAct=finalAct, name="Branch3_output")
        
        ##   combine
        model = Model(inputs=[input1, input2], 
                      outputs = [Branch_1, Branch_2, Branch_3],
                      name = "multi_input_multi_task_DNN")
        
        sgd = SGD(lr=0.01,momentum=0.9, nesterov=True)
        model.compile(optimizer=sgd,loss='categorical_crossentropy',metrics=['accuracy']) 
        
        return model
