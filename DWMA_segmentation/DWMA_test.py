###################################################
#
#   Script to
#   - Calculate prediction of the test dataset
#   - Calculate the parameters to evaluate the prediction
#
##################################################

#Python
import numpy as np
from help_functions import *
from extract_brain_patches import get_data_testing_overlap, recompone_overlap




print('-'*30)
print('Loading test data...')
print('-'*30)



imgs_test = 'your_own_img'
masks_test = 'your_own_mask'


print('-'*30)
print('Preprocessing test data...')
print('-'*30)

##   reshape imgs and masks ; add dimension ====  ##
n_sample = imgs_test.shape[0]
n_slide = imgs_test.shape[1]
n_width = imgs_test.shape[2]
n_height = imgs_test.shape[3]

imgs_test = imgs_test[:, :, np.newaxis, :, :]
masks_test = masks_test[:, :, np.newaxis, :, :]

imgs_test = np.reshape(imgs_test, (n_sample*n_slide, 1, n_width, n_height))
masks_test = np.reshape(masks_test, (n_sample*n_slide, 1, n_width, n_height))

full_img_height = imgs_test.shape[2]
full_img_width = imgs_test.shape[3]

patch_size = 48
patch_height = patch_size
patch_width = patch_size

#the stride in case output with average
stride_height = 24
stride_width = 24
assert (stride_height < patch_height and stride_width < patch_width)


#Loads the data and extracts patches from it

patches_imgs_test, new_height, new_width, masks_test = get_data_testing_overlap(
    test_imgs = imgs_test,  #original
    test_groudTruth = masks_test,  #masks   
    patch_height = patch_height,
    patch_width = patch_width,
    stride_height = stride_height,
    stride_width = stride_width
)



#=========== load the model  =====
n_ch = patches_imgs_test.shape[1]
model = get_unet(n_ch, patch_height, patch_width)
model.load_weights('./Models/model.hdf5')


predictions = model.predict(patches_imgs_test, batch_size=32, verbose=1)
print ("predicted images size :")
print (predictions.shape)


#===== Convert the prediction arrays in corresponding images

pred_patches = sigmoid_pred_to_imgs(predictions, patch_height, patch_width, "original")  #"threshold") #      #  "threshold"



#========== Elaborate and visualize the predicted images ====================
pred_imgs = None
orig_imgs = None
gtruth_masks = None

pred_masks = recompone_overlap(pred_patches, new_height, new_width, stride_height, stride_width)# predictions
gtruth_masks = masks_test  #ground truth masks


pred_masks = pred_masks[:,:,0:full_img_height,0:full_img_width]
gtruth_masks = gtruth_masks[:,:,0:full_img_height,0:full_img_width]

pred_masks  = pred_masks>=0.55

