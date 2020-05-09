import numpy as np
import math

import tensorflow as tf


class Anchors(tf.keras.layers.Layer):
    """Create anchor boxes for Object Detection"""
    
    def __init__(self, size, stride, ratios=None, scales=None, *args, **kwargs):
        """Initiliaze parameters for Anchor Boxes"""
        super(Anchors, self).__init__()
        # strides and sizes align with FPN feature outputs (p2-pn)
#         if not resolution:
#             self.resolution = 640
#         if not sizes:
#             self.sizes = [16, 32, 64, 128, 256]
#         if not strides:
#             self.strides = [4, 8, 16, 32, 64]
        self.size = size
        self.stride = stride
        # ratios and scales applied to all feature levels from FPN output
        if not ratios:
            #self.ratios  = [0.5, 1, 2]
            self.ratios = [1] #used in RetinaFace since faces are typically square-like
        if not scales:
            self.scales  = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]
            
        self.n_anchors = len(self.ratios) * len(self.scales)
            
    def generate_feature_level_base_anchors(self, size):
        """Create K anchors boxes centered on origin for a particular FPN feature level"""
        
        anchors = np.zeros((self.n_anchors, 4)) 
        #scale base size at different scales
        anchors[:, 2:] = size * np.tile(self.scales, (2, len(self.ratios))).T
        # get different combinations of aspect ratios
        areas = anchors[:, 2] * anchors[:, 3]
        anchors[:, 2] = np.sqrt(areas / np.repeat(self.ratios, len(self.scales)))
        anchors[:, 3] = anchors[:, 2] * np.repeat(self.ratios, len(self.scales))
        
        # transform from (x_ctr, y_ctr, w, h) -> (x1, y1, x2, y2)
        anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
        anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T
        
        #self.base_anchors = tf.cast(anchors, dtype=tf.float32)
        return anchors
    
    def shift_and_duplicate(self, anchors, feature_size):
        """Generate bounding boxes by duplicating FPN base anchors every s strides"""
        #feature_size = int(np.round(self.resolution/stride))

        # image_size/stride should equal feature_size (so we could write it for either)
        shift_x = (tf.keras.backend.arange(0, feature_size, dtype=tf.float32) + tf.keras.backend.constant(0.5, dtype=tf.float32)) * self.stride
        shift_y = (tf.keras.backend.arange(0, feature_size, dtype=tf.float32) + tf.keras.backend.constant(0.5, dtype=tf.float32)) * self.stride

        # a tensor that supports cartesian indexing
        shift_x, tf.meshgrid(shift_x)
        shift_y = tf.meshgrid(shift_y)

        # make sure of 1-dimensionsal 
        shift_x = tf.keras.backend.reshape(shift_x, [-1])
        shift_y = tf.keras.backend.reshape(shift_y, [-1])
        
        shifts = tf.keras.backend.stack([shift_x, shift_y, shift_x, shift_y], axis=0)
        shifts = tf.keras.backend.transpose(shifts)
        
        k = tf.keras.backend.shape(shifts)[0]
        
        anchors = tf.convert_to_tensor(anchors, dtype=tf.float32)
        shifts = tf.cast(shifts, dtype=tf.float32)
        

        shifted_anchors = tf.keras.backend.reshape(anchors, [1, self.n_anchors, 4]) + tf.keras.backend.reshape(shifts, [k, 1, 4])
        shifted_anchors = tf.keras.backend.reshape(shifted_anchors, [k * self.n_anchors, 4])
        
        feature_level_anchors = tf.tile(tf.keras.backend.expand_dims(shifted_anchors, axis=0), (feature_size, 1, 1))
        feature_level_anchors = tf.keras.backend.reshape(feature_level_anchors, [(feature_size**2)*self.n_anchors, 4])
        
        return feature_level_anchors
    
    def generate_all_anchors(self):
        """Generate all anchor boxes for every level of the pyramid"""
        self.feature_sizes = [int(np.round(self.resolution/stride)) for stride in self.strides]
        
        all_anchors = [self.generate_feature_level_base_anchors(size=size) for size in self.sizes]
        all_anchors = [self.shift_and_duplicate(layer_anchors, stride) for layer_anchors, stride in zip(all_anchors, self.strides)]
        all_anchors = tf.concat(all_anchors, axis=0)

        return all_anchors
    
    def compute_output_shape(self, input_shape):
        print(input_shape)
        total = np.prod(input_shape[1:3]) * self.n_anchors
        print(total)
        return (input_shape[0], total, 4)
#         if None not in input_shape[1:]:
#             total = np.prod(input_shape[1:3]) * self.n_anchors

#             return (input_shape[0], total, 4)
#         else:
#             return (input_shape[0], None, 4)
    
    def call(self, inputs, **kwargs):
        features = inputs
        features_shape = tf.shape(features)
        batch_size = features_shape[0]
        feature_size = features_shape[1] #reserve first dimension for batch size
        
        anchors = self.generate_feature_level_base_anchors(self.size)
        feature_level_anchors = self.shift_and_duplicate(anchors=anchors, feature_size=feature_size)
        all_anchors = tf.tile(tf.keras.backend.expand_dims(feature_level_anchors, axis=0), (batch_size, 1, 1))
        
        return all_anchors
        
    def get_config(self):
        config = super(Anchors, self).get_config()
        config.update({
            'size'   : self.size,
            'stride' : self.stride,
            'ratios' : self.ratios.tolist(),
            'scales' : self.scales.tolist(),
        })

        return config