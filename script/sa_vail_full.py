# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 21:24:08 2019

@author: zju
"""
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
import numpy as np
import os

model_path = 'DeepTriplet_best_model.h5'

class MyMaskCompute(Layer):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
            
    def call(self, inputs, mask):
        #mask = inputs._keras_mask
        mask = K.cast(mask, K.floatx())
        mask = tf.expand_dims(mask,-1)
        x = inputs * mask
        return x
    
    def compute_mask(self, inputs, mask=None):
        return None

class MySpatialDropout1D(Dropout):

    def __init__(self, rate, **kwargs):
        super(MySpatialDropout1D, self).__init__(rate, **kwargs)
        self.input_spec = InputSpec(ndim=3)

    def _get_noise_shape(self, inputs):
        input_shape = K.shape(inputs)
        noise_shape = (input_shape[0], input_shape[1], 1)
        return noise_shape

def get_model_checkpoint(model_path, verbose=1, save_best_only=True):
    return tf.keras.callbacks.ModelCheckpoint(filepath=model_path,
                                           verbose=verbose,
                                           save_best_only=save_best_only)

def get_callbacks():
    checkpointer = get_model_checkpoint(model_path,
                                        verbose=1,
                                        save_best_only=True)
    return [checkpointer]
  
x_train_1 = np.load('../double_arr_1_b.npy')
x_train_2 = np.load('../double_arr_2_b.npy')
y_train = np.load('../double_arr_3_b.npy')

s_1 = np.load('../double_arr_1_s.npy')
s_2 = np.load('../double_arr_2_s.npy')
s_y = np.load('../double_arr_3_s.npy')

v_1 = np.load(r'random_pair_1.npy')
v_2 = np.load(r'random_pair_2.npy')
v_y = np.load(r'random_pair_y.npy')

def random_arr(arr):
    np.random.seed(5)
    np.random.shuffle(arr)
    np.random.seed(55)
    np.random.shuffle(arr)

random_arr(x_train_1)
random_arr(x_train_2)
random_arr(y_train)
'''
from sklearn.model_selection import KFold, ShuffleSplit

x_a_1 = {}
x_t_1 = {}
x_a_2 = {}
x_t_2 = {}
y_a = {}
y_t_s = {}
kfnum = 0
spl_num = 5
kf = KFold(n_splits=spl_num, shuffle=True, random_state=5)
for train_index, test_index in kf.split(y_train):
    x_a_1[kfnum], x_t_1[kfnum] = x_train_1[train_index], x_train_1[test_index]
    x_a_2[kfnum], x_t_2[kfnum] = x_train_2[train_index], x_train_2[test_index]
    y_a[kfnum], y_t_s[kfnum] = y_train[train_index], y_train[test_index]
    kfnum += 1
'''

x_a_1 = np.concatenate([s_1, x_train_1], 0)
x_a_2 = np.concatenate([s_2, x_train_2], 0)
y_a = np.concatenate([s_y, y_train], 0)
random_arr(x_a_1)
random_arr(x_a_2)
random_arr(y_a)


def main(em_dim, sp_drop, kernel_rate_1, strides_rate_1, kernel_rate_2, strides_rate_2, filter_num_1, filter_num_2, con_drop, fn_drop_1, fn_drop_2, node_num, opti_switch):

    if opti_switch == 0:
        adam = Adam(amsgrad = False)

    elif opti_switch == 1:
        adam = Adam(amsgrad = True)
    
    main_input_a = Input(shape = (1500,), name = 'input_a')
    main_input_b = Input(shape = (1500,), name = 'input_b')

    # tf.dtypes.cast(main_input_a, tf.int32)
    # tf.dtypes.cast(main_input_b, tf.int32)

    embedding_layer = Embedding(25,int(em_dim),mask_zero=True)
    embedded_a = embedding_layer(main_input_a)
    embedded_b = embedding_layer(main_input_b)

    masked_a = MyMaskCompute()(embedded_a)
    masked_b = MyMaskCompute()(embedded_b)

    drop_layer = MySpatialDropout1D(sp_drop)

    dropped_1 = drop_layer(masked_a)
    dropped_2 = drop_layer(masked_b)

    # dropped_1 = masked_a # noting!!!!!!!!!!!!!!!
    # dropped_2 = masked_b # noting!!!!!!!!!!!!!!!

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

    return model

record_max = 10
record_min = 0

model = main(em_dim=15, sp_drop=0.005, kernel_rate_1=0.16, strides_rate_1=0.15, kernel_rate_2=0.14, strides_rate_2=0.25, filter_num_1=150, filter_num_2=175, con_drop=0.05, fn_drop_1=0.2, fn_drop_2=0.1, node_num=256, opti_switch=1)

file_name = 'record_random_pair_val_1.txt' #########################
for n in range(100):
    history_model = model.fit([x_a_1,x_a_2], y_a, batch_size=256, epochs=1, shuffle=True, validation_data=([v_1,v_2], v_y))
    print(str(record_min))

    with open(file_name, 'a') as w:
        w.write(str(n) + ' ')
        w.write('loss: ' + str(history_model.history['loss'][0]) + ' ')
        w.write('acc: ' + str(history_model.history['accuracy'][0]) + ' ')
        w.write('val_loss: ' + str(history_model.history['val_loss'][0]) + ' ')
        w.write('val_acc: ' + str(history_model.history['val_accuracy'][0]) + '\n')

    # if history_model.history['val_loss'][0] < record_max:
    #     record_max = history_model.history['val_loss'][0]
    #     print(str(record_max)+'\tloss_model_saved')
        
    if history_model.history['val_accuracy'][0] > record_min:
        record_min = history_model.history['val_accuracy'][0]
        model.save(r'DeepTrio_acc_random_val_1.h5') ########################
        print(str(record_min)+'\taccuracy_model_saved')

with open(file_name, 'a') as w:
    w.write('Last Value:' + '\n')
    w.write('val_acc: ' + str(record_min) + '\n')