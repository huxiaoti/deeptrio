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
from math import log

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
  
x_train_1 = np.load('data/sa/double_arr_1_b.npy')
x_train_2 = np.load('data/sa/double_arr_2_b.npy')
y_train = np.load('data/sa/double_arr_3_b.npy')

s_1 = np.load('data/sa/double_arr_1_s.npy')
s_2 = np.load('data/sa/double_arr_2_s.npy')
s_y = np.load('data/sa/double_arr_3_s.npy')

def random_arr(arr):
    np.random.seed(5)
    np.random.shuffle(arr)
    np.random.seed(55)
    np.random.shuffle(arr)

random_arr(x_train_1)
random_arr(x_train_2)
random_arr(y_train)

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

for ks in range(spl_num):
    x_a_1[ks] = np.concatenate([s_1, x_a_1[ks]], 0)
    x_a_2[ks] = np.concatenate([s_2, x_a_2[ks]], 0)
    y_a[ks] = np.concatenate([s_y, y_a[ks]], 0)
    random_arr(x_a_1[ks])
    random_arr(x_a_2[ks])
    random_arr(y_a[ks])

def main(em_dim=15, sp_drop=0.005, kernel_rate_1=0.14, strides_rate_1=0.2, kernel_rate_2=0.1, strides_rate_2=0.3, filter_num_1=125, filter_num_2=175, con_drop=0.05, fn_drop_1=0.2, fn_drop_2=0.1, node_num=128, opti_switch=0):

    if opti_switch == 0:
        adam = Adam(amsgrad = False)
        print('^^^^^ False ^^^^^')
    elif opti_switch == 1:
        adam = Adam(amsgrad = True)
        print('^^^^^ True ^^^^^')
    else:
        raise Exception('The format is not in a right way')
    
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

    record_min = 0

    for n in range(100):
        history_model = model.fit([x_a_1[0],x_a_2[0]], y_a[0], batch_size=256, epochs=1, shuffle=True, validation_data=([x_t_1[0],x_t_2[0]], y_t_s[0]))
        if history_model.history['val_accuracy'][0] > record_min:
            record_min = history_model.history['val_accuracy'][0]
            model.save('DeepTriplet_search_' + str(yy) + '.h5') ##########
            print(str(record_min)+'\taccuracy_model_saved')

    with open('search_log.txt', 'a') as log_text:
                
        log_text.write('cycle: ' + str(yy) + '\n')

        log_text.write('em_dim: ' + str(em_dim) + '\n')
        log_text.write('sp_drop: ' + str(sp_drop) + '\n')
        log_text.write('kernel_rate_1: ' + str(kernel_rate_1) + '\n')
        log_text.write('strides_rate_1: ' + str(strides_rate_1) + '\n')
        log_text.write('kernel_rate_2: ' + str(kernel_rate_2) + '\n')
        log_text.write('strides_rate_2: ' + str(strides_rate_2) + '\n')
        log_text.write('filter_num_1: ' + str(filter_num_1) + '\n')
        log_text.write('filter_num_2: ' + str(filter_num_2) + '\n')
        log_text.write('con_drop: ' + str(con_drop) + '\n')
        log_text.write('fn_drop_1: ' + str(fn_drop_1) + '\n')
        log_text.write('fn_drop_2: ' + str(fn_drop_2) + '\n')
        log_text.write('node_num: ' + str(node_num) + '\n')
        log_text.write('opti_switch: ' + str(opti_switch) + '\n')

        log_text.write('accuracy: ' + str(record_min) + '\n')
        log_text.write('-----\n')

    return record_min

bounds = [
          #Discrete 
          {'name':'em_dim','type':'discrete','domain':(10, 15, 20)},
          {'name':'sp_drop','type':'discrete','domain':(0.005, 0.01)},
          {'name':'kernel_rate_1','type':'discrete','domain':(0.12, 0.14, 0.16)},
          {'name':'strides_rate_1','type':'discrete','domain':(0.15, 0.2, 0.25)},
          {'name':'kernel_rate_2','type':'discrete','domain':(0.1, 0.12, 0.14)},
          {'name':'strides_rate_2','type':'discrete','domain':(0.25, 0.3, 0.35)},
          {'name':'filter_num_1','type':'discrete','domain':(100, 125, 150)},
          {'name':'filter_num_2','type':'discrete','domain':(150, 175)},
          {'name':'con_drop', 'type': 'discrete','domain': (0.05, 0.1, 0.15)},
          {'name':'fn_drop_1', 'type': 'discrete','domain': (0.1, 0.2)},
          {'name':'fn_drop_2', 'type': 'discrete','domain': (0.1, 0.2)},
          {'name':'node_num', 'type': 'discrete','domain': (128, 256)},
          #Categorical
          {'name':'opti_switch', 'type': 'categorical','domain': (0, 1)}
         ]

from dotmap import DotMap

def search_param(x):

    opt= DotMap()

    opt.em_dim = float(x[:, 0])
    opt.sp_drop = float(x[:, 1])
    opt.kernel_rate_1 = float(x[:, 2])
    opt.strides_rate_1 = float(x[:, 3])
    opt.kernel_rate_2 = float(x[:, 4])
    opt.strides_rate_2 = float(x[:, 5])
    opt.filter_num_1 = float(x[:, 6])
    opt.filter_num_2 = float(x[:, 7])
    opt.con_drop = float(x[:, 8])
    opt.fn_drop_1 = float(x[:, 9])
    opt.fn_drop_2 = float(x[:, 10])
    opt.node_num = float(x[:, 11])
    opt.opti_switch = int(x[:,12])

    return opt

yy = 35

def f(x):
    
    global yy
    
    local_yy = yy
    local_yy += 1
    yy = local_yy

    opt = search_param(x)
    param = {
            'em_dim':opt.em_dim,
            'sp_drop':opt.sp_drop,
            'kernel_rate_1':opt.kernel_rate_1,
            'strides_rate_1':opt.strides_rate_1,
            'kernel_rate_2':opt.kernel_rate_2,
            'strides_rate_2':opt.strides_rate_2,
            'filter_num_1':opt.filter_num_1,
            'filter_num_2':opt.filter_num_2,
            'con_drop':opt.con_drop,
            'fn_drop_1':opt.fn_drop_1,
            'fn_drop_2':opt.fn_drop_2,
            'node_num':opt.node_num,
            'opti_switch':opt.opti_switch
            }


    result = main(**param)

    evaluation = 1 - result

    # with open('search_log.txt', 'a') as log_text:
    #     log_text.write('evaluation: ' + str(evaluation) + '\n')
    #     log_text.write('---------\n')

    print('cycle: ' + str(yy))
    print('evaluation: ' + str(evaluation))

    return evaluation

import GPy
import GPyOpt

opt_model = GPyOpt.methods.BayesianOptimization(f=f, domain=bounds, initial_design_numdata=10)
opt_model.run_optimization(max_iter=50)

with open('search_log.txt', 'a') as log_text:
    log_text.write('result: \n')

    for i,v in enumerate(bounds):
        name = v['name']
        log_text.write('parameter {}: {}\n'.format(name,opt_model.x_opt[i]))
    log_text.write('evaluation: ' + str(1 - opt_model.fx_opt) + '\n')
