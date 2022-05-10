# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 21:24:08 2019

@author: zju
"""
import os
import argparse
import numpy as np
import pickle as pkl
from sklearn.model_selection import KFold, ShuffleSplit

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam

from build_my_layer import MyMaskCompute, MySpatialDropout1D
from utility import random_arr, array_split
from input_preprocess import preprocess


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--fold_index', type=int, default=0)
parser.add_argument('--epoch', type=int, default=100, help='the maximum number of epochs')
parser.add_argument('--outer_product', default=False, action='store_true', help='Whether apply max-pooling on outer-product of two proteins')

args = parser.parse_args()

if not args.outer_product:
    no_product_max = True
else:
    no_product_max = False

pair_file = args.dataset + '_pair.tsv'
seq_file = args.dataset + '_seq.tsv'


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

x_train_1, x_train_2, y_train, single_1, single_2, single_y, fix_text = preprocess(pair_file, seq_file)

random_arr(x_train_1)
random_arr(x_train_2)
random_arr(y_train)
random_arr(fix_text)

x_1, x_2, y_train, v_1, v_2, v_y, h_train, h_validation = array_split(5, x_train_1, x_train_2, y_train, single_1, single_2, single_y, fix_text)

    
main_input_a = Input(shape = (1500,), name = 'input_a')
main_input_b = Input(shape = (1500,), name = 'input_b')

embedding_layer = Embedding(25, 15, mask_zero=True)
embedded_a = embedding_layer(main_input_a)
embedded_b = embedding_layer(main_input_b)

masked_a = MyMaskCompute()(embedded_a)
masked_b = MyMaskCompute()(embedded_b)

drop_layer = MySpatialDropout1D(args.sp_drop)

dropped_1 = drop_layer(masked_a)
dropped_2 = drop_layer(masked_b)

tensor = []

for n in range(2, 35):
    
    if n <= 15:
        conv_layer = Conv1D(filters=150,
                            kernel_size = int(np.ceil(0.16 * n**2)),
                            padding = 'valid',
                            activation = 'relu',
                            use_bias= False,
                            strides = int(np.ceil(0.15 * (n-1))))
    else:
        conv_layer = Conv1D(filters = 175,
                            kernel_size = int(np.ceil(0.14 * n**2)),
                            padding = 'valid',
                            activation = 'relu',
                            use_bias= False,
                            strides = int(np.ceil(0.25 * (n-1))))
    
    conv_out_1 = conv_layer(dropped_1)
    conv_out_2 = conv_layer(dropped_2)

    conv_out_1 = SpatialDropout1D(args.con_drop)(conv_out_1)
    conv_out_2 = SpatialDropout1D(args.con_drop)(conv_out_2)
    
    max_layer = MaxPooling1D(pool_size=int(conv_layer.output_shape[1]))
    
    pool_out_1 = max_layer(conv_out_1)
    pool_out_2 = max_layer(conv_out_2)

    flat_out_1 = Flatten()(pool_out_1)
    flat_out_2 = Flatten()(pool_out_2)

    if args.no_product_max:

        pool_out = flat_out_1 + flat_out_2

    else:

        flat_out_1 = tf.reshape(flat_out_1, (-1, int(conv_layer.output_shape[2]), 1))
        flat_out_2 = tf.reshape(flat_out_2, (-1, 1, int(conv_layer.output_shape[2])))

        pool_out = tf.matmul(flat_out_1, flat_out_2)

        flat_out = 1/2 * (tf.reduce_max(pool_out, axis=1) + tf.reduce_max(pool_out, axis=2))
    
    tensor.append(flat_out)
    
concatenated = Concatenate()(tensor)

x = Dropout(0.2)(concatenated)
x = Dense(args.node_num)(x)
x = Dropout(0.1)(x)
x = Activation('relu')(x)
x = Dense(3)(x)

main_output = Activation('softmax', name = 'out')(x)

adam = Adam(amsgrad = True)
 
model = Model(inputs = [main_input_a,main_input_b], outputs = main_output)

model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])


acc_max = 0
loss_min = 100

file_name = 'DeepTrio_{}_{}_{}'.format(args.seed, args.fold_index, args.dataset)
loss_values = []
acc_values = []
for i in range(args.epoch):
    history_model = model.fit([x_1[args.fold_index],x_2[args.fold_index]], y_train[args.fold_index], batch_size=256, epochs=1, shuffle=True, validation_data=([v_1[args.fold_index],v_2[args.fold_index]], v_y[args.fold_index]))
    loss_values.append(history_model.history['val_loss'][0])
    acc_values.append(history_model.history['val_accuracy'][0])
    
    if acc_values[-1] > acc_max:
        acc_max = acc_values[-1]
        model.save(file_name + '_acc.h5')
        best_epoch = i + 1
        
    if loss_values[-1] < loss_min:
        loss_min = loss_values[-1]
        model.save(file_name + '_loss.h5')
        best_epoch = i + 1
        
    print('This is {} epoch. Best epoch is {}, loss is {}, acc is {}'.format((i+1), best_epoch, loss_values[best_epoch - 1], acc_values[best_epoch - 1]))

model_loss = tf.keras.models.load_model(file_name + '_loss.h5', custom_objects={'MyMaskCompute':MyMaskCompute, 'MySpatialDropout1D':MySpatialDropout1D})
preds_loss = model_loss.predict([v_1[args.fold_index], v_2[args.fold_index]])

model_acc = tf.keras.models.load_model(file_name + '_acc.h5', custom_objects={'MyMaskCompute':MyMaskCompute, 'MySpatialDropout1D':MySpatialDropout1D})
preds_acc = model_acc.predict([v_1[args.fold_index], v_2[args.fold_index]])


with open(('predloss_predacc_label_' + file_name + '.pkl'),'wb') as fw:
    pkl.dump([preds_loss, preds_acc, h_validation[args.fold_index]], fw)
