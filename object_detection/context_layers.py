import tensorflow as tf


class FeaturePyramid(tf.keras.layers.Layer):
    """Builds the Feature Pyramid with the feature maps from the backbone.

    Arguments:
      num_classes (int): Number of classes in the dataset.
      backbone (str): The backbone to build the feature pyramid from.
    """

    def __init__(self, backbone=None, **kwargs):
        super(FeaturePyramid, self).__init__(name="FeaturePyramid", **kwargs)
        self.backbone = load_backbone()

        self.p2_lateral_conv = Conv2D(256, kernel_size=(1, 1), strides=1, padding='same', activation=None, name='p2_lateral_conv')
        self.p2_lc_reshape = tf.keras.layers.Reshape((160, 160, 256))
        self.p3_upsample =  UpSampling2D(size=(2, 2), name='p3_upsampled')
        self.p2_add = Add(name='p2_add')
        
        self.p3_lateral_conv = Conv2D(256, kernel_size=(1, 1), strides=1, padding='same', activation=None, name='p3_lateral_conv')
        self.p3_lc_reshape = tf.keras.layers.Reshape((80, 80, 256))
        self.p4_upsample =  UpSampling2D(size=(2, 2), name='p4_upsampled')
        self.p3_add = Add(name='p3_add')
        
        self.p4_lateral_conv = Conv2D(256, kernel_size=(1, 1), strides=1, padding='same', activation=None, name='p4_lateral_conv')
        self.p4_lc_reshape = tf.keras.layers.Reshape((40, 40, 256))
        self.p5_upsample =  UpSampling2D(size=(2, 2), name='p5_upsampled')
        self.p4_add = Add(name='p4_add')
        
        self.p5_lateral_conv = Conv2D(256, kernel_size=(1, 1), strides=1, padding='same', activation=None, name='p5_lateral_conv')
        # glorot same as xavier
        self.p6_lateral_conv = Conv2D(256, kernel_size=(3, 3), strides=2, padding='same', activation=None, kernel_initializer='glorot_normal', name='p6_lateral_conv')
    
        self.p4_conv_out =  Conv2D(256, kernel_size=(3, 3), strides=1, padding='same', activation=None, name='p4_conv_out')
        self.p3_conv_out = Conv2D(256, kernel_size=(3, 3), strides=1, padding='same', activation=None, name='p3_conv_out')
        self.p2_conv_out = Conv2D(256, kernel_size=(3, 3), strides=1, padding='same', activation=None, name='p2_conv_out')

    def call(self, images, training=False):
        c4, c3, c2, c5 = self.backbone(images, training=training)
        p6 = self.p6_lateral_conv(c5)
        p5 = self.p5_lateral_conv(c5)
        
        p4_lateral_conv = self.p4_lateral_conv(c4)
        p5_upsampled = self.p5_upsample(p5)
        p4_add = self.p4_add([p4_lateral_conv, p5_upsampled])
        p4 = self.p4_conv_out(p4_add)
        
        p3_lateral_conv = self.p3_lateral_conv(c3)
        p4_upsampled = self.p4_upsample(p4)
        p3_add = self.p3_add([p3_lateral_conv, p4_upsampled])
        p3 = self.p3_conv_out(p3_add)
        
        #named the same thing in ResNet50 and ResNet152
        p2_lateral_conv = self.p2_lateral_conv(c2)
        p3_upsampled = self.p3_upsample(p3)
        p2_add = self.p2_add([p2_lateral_conv, p3_upsampled])
        p2 = self.p2_conv_out(p2_add)
        return p2, p3, p4, p5, p6
    
    

class ClassificationSubnet(tf.keras.layers.Layer):
    """Classification Subnet on top of FPN
    
    Arguments:
        num_classes (int): Number of classes
        num_base_anchors (int): Number of base anchors (# strides * # aspect ratios)
        prior (float): Prior probability of anchor box containing an object (default=0.1)
    """
    
    def __init__(self, n_classes, n_base_anchors, prior=0.01):
        super(ClassificationSubnet, self).__init__()
        self.K = n_classes
        self.A = n_base_anchors
        self.kernel_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)
        self.bias_intializer = tf.constant_initializer(-np.log((1 - prior) / prior))
        
        self.feature_class_conv = tf.keras.layers.Conv2D(256, 
                                                         kernel_size=(3, 3), 
                                                         strides=1,
                                                         padding='same',
                                                         kernel_initializer=self.kernel_initializer,
                                                         bias_initializer='zeros',
                                                         activation='relu')
        self.class_subnet_conv_out =  tf.keras.layers.Conv2D(self.K*self.A,
                                                             kernel_size=(3, 3),
                                                             strides=1,
                                                             padding='same',
                                                             kernel_initializer=self.kernel_initializer,
                                                             bias_initializer=self.prior,
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
    
    def __init__(self, n_landmarks, n_base_anchors):
        super(RegressionSubnet, self).__init__()
        self.n_landmarks = n_landmarks
        self.A = num_base_anchors
        self.kernel_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)
        
        self.regression_feature_conv = tf.keras.layers.Conv2D(256, 
                                                              kernel_size=(3, 3), 
                                                              strides=1,
                                                              padding='same',               
                                                              kernel_initializer=self.kernel_initializer,
                                                              bias_initializer='zeros',
                                                              activation='relu')
        self.regression_conv_out = tf.keras.layers.Conv2D(self.n_landmarks*self.A,
                                                          kernel_size=(3, 3),
                                                          strides=1,
                                                          padding='same',
                                                          kernel_initializer=self.kernel_initializer,
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