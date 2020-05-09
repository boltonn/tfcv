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
        
    def call(self, inputs):
        outputs = inputs
        for i in range(4):
                outputs = Conv2D(256, 
                                 kernel_size=(3, 3), 
                                 strides=1,
                                 padding='same',
                                 kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None),
                                 bias_initializer='zeros',
                                 activation='relu')(outputs)
        
        outputs = Conv2D(self.K*self.A,
                         kernel_size=(3, 3),
                         strides=1,
                         padding='same',
                         kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None),
                         bias_initializer=PriorProbability(probability=self.prior),
                         activation='relu')(outputs)  # (w x h x C) - (w x h x (K*A))
        
        # K anchors at each center pixel (width*height)
        outputs = tf.keras.layers.Reshape((-1, self.K))(outputs) # (w x h x (K*A)) - (w*h*A, K) 
        outputs = tf.keras.layers.Activation('sigmoid')(outputs)
        
        return outputs
    
    
class RegressionSubnet(tf.keras.layers.Layer):
    """Regression Subnet on top of FPN"""
    
    def __init__(self, n_landmarks, A):
        super(RegressionSubnet, self).__init__()
        self.n_landmarks = n_landmarks
        self.A = A
        
    def call(self, inputs):
        outputs = inputs
        for i in range(4):
                outputs = Conv2D(256, 
                                 kernel_size=(3, 3), 
                                 strides=1,
                                 padding='same',
                                 kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None),
                                 bias_initializer='zeros',
                                 activation='relu')(outputs)
        
        outputs = Conv2D(self.n_landmarks*self.A,
                         kernel_size=(3, 3),
                         strides=1,
                         padding='same',
                         kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None),
                         bias_initializer='zeros',
                         activation='relu')(outputs)  # (w x h x C) - (w x h x (K*A))
        
        # K anchors at each center pixel (width*height)
        outputs = tf.keras.layers.Reshape((-1, self.n_landmarks))(outputs) # (w x h x (n_landmarks*A)) - (w*h*n_landmarks, K) 
        
        return outputs