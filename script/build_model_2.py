import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
import numpy as np
from custom_layer import MyMaskCompute, MySpatialDropout1D


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