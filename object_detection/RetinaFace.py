import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Concatenate, UpSampling2D, Add

from .context_layers import ClassificationSubnet, RegressionSubnet
    

class RetinaFace(tf.keras.Model):
    """RetinaFace w/o G-CNN"""
    def __init__(self, input_shape=(640, 640, 3)):
        super(RetinaFace, self).__init__()
        
        self.backbone_model = tf.keras.applications.ResNet152(weights='imagenet', 
                                                              input_shape=input_shape,
                                                              include_top=False)
        self.backbone_model.trainable = False # freeze all layers of ResNet
        self.p2_lateral_conv = Conv2D(256, kernel_size=(1, 1), strides=1, padding='same', activation=None, name='p2_lateral_conv')
        self.p3_lateral_conv = Conv2D(256, kernel_size=(1, 1), strides=1, padding='same', activation=None, name='p3_lateral_conv')
        self.p4_lateral_conv = Conv2D(256, kernel_size=(1, 1), strides=1, padding='same', activation=None, name='p4_lateral_conv')
        self.p5_lateral_conv = Conv2D(256, kernel_size=(1, 1), strides=1, padding='same', activation=None, name='p5_lateral_conv')
        # glorot same as xavier
        self.p6_lateral_conv = Conv2D(256, kernel_size=(3, 3), strides=2, padding='same', activation=None, kernel_initializer='glorot_normal', name='p6_lateral_conv')
    
        
    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        x = self.backbone_model(inputs)
        p6 = self.p6_lateral_conv(x)
        p5 = self.p5_lateral_conv(x)
        
        p4_lateral_conv = self.p4_lateral_conv(self.backbone_model.get_layer('conv4_block36_out').output)
        p4_lateral_conv = tf.reshape(p4_lateral_conv, [batch_size, 40, 40, 256])
        p5_upsampled = UpSampling2D(size=(2, 2), name='p5_upsampled')(p5)
        p4_add = Add(name='p4_add')([p4_lateral_conv, p5_upsampled])
        p4 = Conv2D(256, kernel_size=(3, 3), strides=1, padding='same', activation=None, name='p4_conv_out')(p4_add)
        
        p3_lateral_conv = self.p3_lateral_conv(self.backbone_model.get_layer('conv3_block8_out').output)
        p3_lateral_conv = tf.reshape(p3_lateral_conv, [batch_size, 80, 80, 256])
        p4_upsampled = UpSampling2D(size=(2, 2), name='p4_upsampled')(p4)
        p3_add = Add(name='p3_add')([p3_lateral_conv, p4_upsampled])
        p3 = Conv2D(256, kernel_size=(3, 3), strides=1, padding='same', activation=None, name='p3_conv_out')(p3_add)
        
        p2_lateral_conv = self.p2_lateral_conv(self.backbone_model.get_layer('conv2_block3_out').output)
        p2_lateral_conv = tf.reshape(p2_lateral_conv, [batch_size, 160, 160, 256])
        p3_upsampled = UpSampling2D(size=(2, 2), name='p3_upsampled')(p3)
        p2_add = Add(name='p2_add')([p2_lateral_conv, p3_upsampled])
        p2 = Conv2D(256, kernel_size=(3, 3), strides=1, padding='same', activation=None, name='p2_conv_out')(p2_add)
        
        features = [p2, p3, p4, p5, p6]
        
        #return features

        # K=2 since face or not face
        classification_outputs = []
        for feature_layer in features:
            classification_outputs.append(ClassificationSubnet(K=2, A=3, prior=.01)(feature_layer))
            
        # bounding box regression
        regression_outputs = []
        for feature_layer in features:
            regression_outputs.append(RegressionSubnet(n_landmarks=4, A=3)(feature_layer))
            
#         # facial landmark regression
#         landmarks_outputs = []
#         for feature_layer in features:
#             landmarks_outputs.append(RegressionSubnet(n_landmarks=5, A=3)(feature_layer))
        #return features    
        return classification_outputs, regression_outputs