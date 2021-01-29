
import numpy as np
from extract_brain_patches import get_data_training
from help_functions import *





print('-'*30)
print('Loading and preprocessing train data...')
print('-'*30)

imgs_data = 'your own imgs'
masks_data = 'your own masks'

print('-'*30)
print('Splitting into train and testing data...')
print('-'*30)

N_train = 49
imgs_train = imgs_data[N_train:, :,:,:]
masks_train = masks_data[N_train:, :,:,:]


print('-'*30)
print('Preprocessing train data...')
print('-'*30)


##   reshape imgs and masks ; add dimension ====  ##
n_sample = imgs_train.shape[0]
n_slide = imgs_train.shape[1]
n_width = imgs_train.shape[2]
n_height = imgs_train.shape[3]

imgs_train = np.reshape(imgs_train, (n_sample*n_slide, n_width, n_height))
masks_train = np.reshape(masks_train, (n_sample*n_slide, n_width, n_height))

imgs_train = imgs_train[:,np.newaxis, :, :]
masks_train = masks_train[:,np.newaxis, :, :]

#============ This function extracts patches from the training samples and save =======#
N_patch_per_img = 50
patch_size = 48

patches_imgs_train, patches_masks_train = get_data_training(
    train_imgs = imgs_train,
    train_groudTruth = masks_train,
    train_bordermasks = imgs_train,
    patch_height = patch_size,
    patch_width = patch_size,
    N_subimgs_per_img = N_patch_per_img,
    inside_FOV = True #select the patches only inside the FOV  (default == True)
)


np.save('./Processed_patch_data/.....npy', patches_imgs_train)
np.save('./Processed_patch_data/.....npy', patches_masks_train)



