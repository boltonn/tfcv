import tensorflow as tf
from tensorflow.keras.layers import Conv2D

from initializers import PriorProbability

class ClassificationSubnet(tf.keras.layers.Layer):
    """Classification Subnet on top of FPN"""
    
    def __init__(self, K, A, prior):
        super(ClassificationSubnet, self).__init__()
        self.K = K
        self.A = A
        self.prior = prior
        
        self.feature_class_conv = Conv2D(256, 
                                   kernel_size=(3, 3), 
                                   strides=1,
                                   padding='same',
                                   kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None),
                                   bias_initializer='zeros',
                                   activation='relu')
        self.class_subnet_conv_out =  Conv2D(self.K*self.A,
                                             kernel_size=(3, 3),
                                             strides=1,
                                             padding='same',
                                             kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None),
                                             bias_initializer=PriorProbability(probability=self.prior),
                                             activation='relu')
        
        
    def compute_output_shape(self, input_shape):
        return input_shape[0]
        
    def call(self, inputs):
        outputs = inputs
        for i in range(4):
                outputs = self.feature_class_conv(outputs)
        
        outputs = self.class_subnet_conv_out(outputs)  # (w x h x C) - (w x h x (K*A))
        
        # K anchors at each center pixel (width*height)
        outputs = tf.keras.layers.Reshape((-1, self.K))(outputs) # (w x h x (K*A)) - (w*h*A, K) 
        outputs = tf.keras.layers.Activation('sigmoid')(outputs)
        
        return outputs
        
    def get_config(self):
        config = super(ClassificationSubnet, self).get_config()
        config.update({
            'K'   : self.K,
            'A' : self.A,
            'prior' : self.prior
        })

        return config
    
    
class RegressionSubnet(tf.keras.layers.Layer):
    """Regression Subnet on top of FPN"""
    
    def __init__(self, n_landmarks, A):
        super(RegressionSubnet, self).__init__()
        self.n_landmarks = n_landmarks
        self.A = A
        
        self.regression_feature_conv = Conv2D(256, 
                                              kernel_size=(3, 3), 
                                              strides=1,
                                              padding='same',
                                              kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None),
                                              bias_initializer='zeros',
                                              activation='relu')
        self.regression_conv_out = Conv2D(self.n_landmarks*self.A,
                                          kernel_size=(3, 3),
                                          strides=1,
                                          padding='same',
                                          kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None),
                                          bias_initializer='zeros',
                                          activation='relu')
        
    def compute_output_shape(self, input_shape):
        return input_shape[0]
        
    def call(self, inputs):
        outputs = inputs
        for i in range(4):
                outputs = self.regression_feature_conv(outputs)
        
        outputs = self.regression_conv_out(outputs)  # (w x h x C) - (w x h x (K*A))
        
        # K anchors at each center pixel (width*height)
        outputs = tf.keras.layers.Reshape((-1, self.n_landmarks))(outputs) # (w x h x (n_landmarks*A)) - (w*h*n_landmarks, K) 
        
        return outputs
    
    def get_config(self):
        config = super(RegressionSubnet, self).get_config()
        config.update({
            'n_landmarks'   : self.n_landmarks,
            'A': self.A
        })