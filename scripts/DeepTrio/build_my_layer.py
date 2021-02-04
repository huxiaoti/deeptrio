import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
import numpy as np

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