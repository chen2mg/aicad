###################################################
#
#   Script to:
#   - Load the images and extract the patches
#   - Define the neural network
#   - perform the training
#
##################################################


import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
from Models.Unet_models import get_unet


print ("Load training patch data")
patches_imgs_train = np.load("./Processed_patch_data/.....npy")
patches_masks_train = np.load("./Processed_patch_data/....npy") 


#=========== Construct and save the model architecture =====
n_ch = patches_imgs_train.shape[1]
patch_height = patches_imgs_train.shape[2]
patch_width = patches_imgs_train.shape[3]



print ("Constructing UNet")
model = get_unet(n_ch, patch_height, patch_width)  #the U-net model


#============  compilation of U-net ==================================

print ("Check: final output of the network:")
print (model.output_shape)

#============  Training ==================================
#training settings

print('-'*30)
print('Fitting model...')
print('-'*30)

N_epochs = 20
batch_size = 128
model.fit(patches_imgs_train, patches_masks_train, 
          epochs=N_epochs, 
          batch_size=batch_size, 
          verbose=1, 
          shuffle= True, 
          validation_split=0.2)



##========== Save and test the last model ===================
model.save('./Models/model.hdf5', overwrite=True)


