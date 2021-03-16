# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 21:24:08 2019

@author: zju
"""
import os
from math import log
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import KFold, ShuffleSplit
from build_my_layer import MyMaskCompute, MySpatialDropout1D
from utility import random_arr, array_split
from input_preprocess import preprocess
import warnings
import argparse


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser(description='train DeepTrio')
parser.add_argument('-p', '--ppi', required=True, type=str, help='configuration of the PPI file, which contains the protein 1 id, protein 2 id and the class they belong to, and they are splited by table key')
parser.add_argument('-d', '--database', required=True, type=str, help='configuration of the protein sequence database, which contains the protein id and its sequence, and they are splited by table key')
parser.add_argument('-e', '--epoch', default='100', type=str, help='the maximum number of epochs')

static_args = parser.parse_args()
file_1_path = static_args.ppi
file_2_path = static_args.database
epoch_number = int(static_args.epoch)

print('\nWelcome to use our tool')
print('\nVersion: 1.0.0')
print('\nAny problem, please contact mchen@zju.edu.cn')
print('\nStart to process the raw data')

x_train_1, x_train_2, y_train, single_1, single_2, single_y, _dict, _protein_list = preprocess(file_1_path, file_2_path)

random_arr(x_train_1)
random_arr(x_train_2)
random_arr(y_train)

x_1, x_2, y_train, v_1, v_2, v_y = array_split(5, x_train_1, x_train_2, y_train, single_1, single_2, single_y, _dict, _protein_list)

print('\nStart to train DeepTrio model')
print('\nAfter training, you may select the best model manually according to the recording file')

em_dim=15
sp_drop=0.005
kernel_rate_1=0.16
strides_rate_1=0.15
kernel_rate_2=0.14
strides_rate_2=0.25
filter_num_1=150
filter_num_2=175
con_drop=0.05
fn_drop_1=0.2
fn_drop_2=0.1
node_num=256
opti_switch=1

if opti_switch == 0:
    adam = Adam(amsgrad = False)
    # print('^^^^^ False ^^^^^')
elif opti_switch == 1:
    adam = Adam(amsgrad = True)
    # print('^^^^^ True ^^^^^')
else:
    raise Exception('The format is not in a right way')

main_input_a = Input(shape = (2000,), name = 'input_a')
main_input_b = Input(shape = (2000,), name = 'input_b')

embedding_layer = Embedding(25,int(em_dim),mask_zero=True)
embedded_a = embedding_layer(main_input_a)
embedded_b = embedding_layer(main_input_b)

masked_a = MyMaskCompute()(embedded_a)
masked_b = MyMaskCompute()(embedded_b)

drop_layer = MySpatialDropout1D(sp_drop)

dropped_1 = drop_layer(masked_a)
dropped_2 = drop_layer(masked_b)

tensor = []

for n in range(2,35):
    
    if n <= 15:
        conv_layer = Conv1D(filters= int(filter_num_1),
        kernel_size = int(np.ceil(kernel_rate_1 * n**2)),
        padding = 'valid',
        activation = 'relu',
        use_bias= False,
        strides = int(np.ceil(strides_rate_1*(n-1))))
    else:
        conv_layer = Conv1D(filters= int(filter_num_2),
        kernel_size = int(np.ceil(kernel_rate_2 * n**2)),
        padding = 'valid',
        activation = 'relu',
        use_bias= False,
        strides = int(np.ceil(strides_rate_2*(n-1))))
    
    conv_out_1 = conv_layer(dropped_1)
    conv_out_2 = conv_layer(dropped_2)

    conv_out_1 = SpatialDropout1D(con_drop)(conv_out_1)
    conv_out_2 = SpatialDropout1D(con_drop)(conv_out_2)
    
    max_layer = MaxPooling1D(pool_size=int(conv_layer.output_shape[1]))
    
    pool_out_1 = max_layer(conv_out_1)
    pool_out_2 = max_layer(conv_out_2)
    
    pool_out = pool_out_1 + pool_out_2
    
    flat_out = Flatten()(pool_out)
    
    tensor.append(flat_out)
    
concatenated = Concatenate()(tensor)
x = Dropout(fn_drop_1)(concatenated)
x = Dense(int(node_num))(x)
x = Dropout(fn_drop_2)(x)
x = Activation('relu')(x)
x = Dense(3)(x)
main_output = Activation('softmax', name = 'out')(x)

model = Model(inputs = [main_input_a,main_input_b], outputs = main_output)

model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

record_min = 0
sig = ['zero','first','second','third','forth']
ix = 2 ######################

for n in range(epoch_number):
    history_model = model.fit([x_1[ix],x_2[ix]], y_train[ix], batch_size=256, epochs=1, shuffle=True, validation_data=([v_1[ix],v_2[ix]], v_y[ix]))
    if history_model.history['val_accuracy'][0] > record_min:
        record_min = history_model.history['val_accuracy'][0]
        model.save('DeepTrio_acc_' + sig[ix] + '.h5') ##########
        print(str(record_min)+'\taccuracy_model_saved')
    print('current_accuracy: ' + str(record_min))
    with open('record_trio_' + sig[ix] + '.txt', 'a') as w: ##################
        w.write(str(n) + '\t')
        w.write('loss: ' + str(history_model.history['loss'][0]) + '\t')
        w.write('acc: ' + str(history_model.history['accuracy'][0]) + '\t')
        w.write('val_loss: ' + str(history_model.history['val_loss'][0]) + '\t')
        w.write('val_acc: ' + str(history_model.history['val_accuracy'][0]) + '\n')

    