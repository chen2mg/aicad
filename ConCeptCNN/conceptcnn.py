# Ming Chen
import numpy as np
import scipy.io as sio
import os
import math
# plot the model
os.environ["PATH"] += os.pathsep + 'C:\\Program Files (x86)\\Graphviz2.38\\bin'
from keras.utils.vis_utils import plot_model

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, accuracy_score
from keras.layers import Dense, Dropout, BatchNormalization, Activation, Input, Conv2D, Flatten, MaxPooling2D, GlobalAveragePooling2D, AveragePooling2D, concatenate
from keras.models import Sequential, Model

import keras

from indeption_model import inception_module
import scipy.io as sio
import jason
# import matplotlib.pyplot as plt
from model import build_fc
from model import check_models

def load_dti_data():
    dti = np.loadtxt(os.path.join('data', 'dti', 'DTI_FA_matrix.npy'))
    data = dti[:, 1:]
    y = dti[:, 0]
    if y.sum() > 100:
        y = np.where(dti[:, 0] >= 90, 0, 1)
    return data, y


def load_adhd_data(name):
    # adhd = sio.loadmat(os.path.join('data', 'adhd', 'aal', 'aal.mat'))
    # train_set = adhd['ADHD200_AAL_TCs_filtfix']
    # holdout_set = adhd['ADHD200_AAL_TCs_TestRelease']
    # all_set = np.concatenate((train_set, holdout_set), axis=0)
    #
    # data = all_set[:, 2:]
    # y = np.where(all_set[:, 1] == 0.0, 0, 1)

    # load holdout data
    test = sio.loadmat(os.path.join('data', 'adhd', 'adhd_test.mat'))
    fmri_test = test['fmri']
    y_test = test['adhd']
    y_test = np.where(y_test>0, 1, 0).flatten()
    # load train data
    train = sio.loadmat(os.path.join('data', 'adhd', 'adhd_train.mat'))
    fmri_train = train['fmri']
    y_train = train['adhd']
    y_train = np.where(y_train > 0, 1, 0).flatten()

    if name == 'test':
        return fmri_test, y_test
    elif name == 'train':
        return fmri_train, y_train
    else:
        fmri = np.concatenate((fmri_train, fmri_test), axis=0)
        y = np.concatenate([y_train, y_test])
        return fmri, y
        
        
def run(batch_size=4, epochs=12, save_csv=False):
    data, y = load_adhd_data('all')
    area = data.shape[1]
    ACC =[]
    SENSI =[]
    SPEC =[]
    AUC =[]

    kf = StratifiedKFold(n_splits=3, random_state=10, shuffle=True)
    fold = 1
    for train_index, test_index in kf.split(data, y):
        print('\nfold:', fold)
        fold = fold + 1
        col_train, col_test = data[train_index], data[test_index]
        y_train, y_test = y[train_index], y[test_index]

        col_train = col_train.reshape(-1, area, area, 1)
        col_test = col_test.reshape(-1, area, area, 1)

        model = build_fc(input_shape=(area, area, 1))
        #model.summary()
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(col_train, y_train, batch_size=batch_size, validation_data=(col_test, y_test), epochs=epochs, verbose=1)
        scores = model.evaluate(col_test, y_test)
        pred = model.predict(col_test)
        pred = np.where(pred > 0.5, 1, 0)

        matrix = confusion_matrix(y_test, pred)
        roc_curve = roc_auc_score(y_test, pred)
        # matrix = confusion_matrix(y_test.argmax(axis=1), pred.argmax(axis=1))
        # roc_curve = roc_auc_score(y_test.argmax(axis=1), pred.argmax(axis=1))
        specificity = matrix[0, 0] / (matrix[0, 0] + matrix[0, 1])
        sensitivity = matrix[1, 1] / (matrix[1, 1] + matrix[1, 0])
        print('Test loss:', scores[0])
        print('Confusion matrix:\n', matrix)
        print('Test accuracy:', scores[1])
        print('Test specificity', specificity)
        print('Test sensitivity', sensitivity)
        print('roc_curve', roc_curve)
        ACC.append(scores[1])
        SENSI.append(sensitivity)
        SPEC.append(specificity)
        AUC.append(roc_curve)

    print('\n\n\n')
    print('Mean accuracy:', np.mean(ACC))
    print('Mean specificity', np.mean(SPEC))
    print('Mean sensitivity', np.mean(SENSI))
    print('Mean roc_curve', np.mean(AUC))
    #return model
    if save_csv:
        with open('adhd_5_fold_cnn.csv', 'a') as fd:
            fd.write(str(np.mean(ACC)) + '; ' + str(np.mean(SPEC)) + '; ' + str(np.mean(SENSI)) + '; '
                     + str(np.mean(AUC)) + ';\n')
                     

if __name__ == '__main__':
    #model = run(batch_size=16, epochs=1, save_csv=False)
    #plot_model(model, 'model_final.png')
    #check_models()

    for i in range(50):
        print('iteration:', i+1)
        run_multi_filter(batch_size=4, epochs=500, save_csv=True)
