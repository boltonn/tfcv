import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Concatenate, UpSampling2D, Add

from context_layers import ClassificationSubnet, RegressionSubnet
    

#model subclassing not working b/c Add layer doesnt catch shape changes
tf.keras.backend.clear_session() 

class RetinaFace(tf.keras.Model):
    def __init__(self, input_shape=(640, 640, 3), backbone='resnet50'):
        super(RetinaFace, self).__init__()
        self.backbone_name = backbone
        if self.backbone_name == 'resnet50':
            self.backbone_model = tf.keras.applications.ResNet50(weights='imagenet', 
                                                                 input_shape=input_shape,
                                                                 include_top=False)
        elif self.backbone_name == 'resnet152':
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
        
        if self.backbone_name=='resnet50':
            p4_lateral_conv = self.p4_lateral_conv(self.backbone_model.get_layer('conv4_block6_out').output)
        elif self.backnone_name=='resnet152':
            p4_lateral_conv = self.p4_lateral_conv(self.backbone_model.get_layer('conv4_block36_out').output)
        else:
            raise ValueError("Backbone model must be one of 'resnet50' or 'resnet152'\n conv4 block layer not found")
        p4_lateral_conv = tf.reshape(p4_lateral_conv, [batch_size, 40, 40, 256])
        p5_upsampled = UpSampling2D(size=(2, 2), name='p5_upsampled')(p5)
        p4_add = Add(name='p4_add')([p4_lateral_conv, p5_upsampled])
        p4 = Conv2D(256, kernel_size=(3, 3), strides=1, padding='same', activation=None, name='p4_conv_out')(p4_add)
        
        if self.backbone_name=='resnet50':
            p3_lateral_conv = self.p3_lateral_conv(self.backbone_model.get_layer('conv3_block4_out').output)
        elif self.backnone_name=='resnet152':
            p3_lateral_conv = self.p3_lateral_conv(self.backbone_model.get_layer('conv3_block8_out').output)
        else:
            raise ValueError("Backbone model must be one of 'resnet50' or 'resnet152'\n conv3 block layer not found")
        p3_lateral_conv = tf.reshape(p3_lateral_conv, [batch_size, 80, 80, 256])
        p4_upsampled = UpSampling2D(size=(2, 2), name='p4_upsampled')(p4)
        p3_add = Add(name='p3_add')([p3_lateral_conv, p4_upsampled])
        p3 = Conv2D(256, kernel_size=(3, 3), strides=1, padding='same', activation=None, name='p3_conv_out')(p3_add)
        
        #named the same thing in ResNet50 and ResNet152
        p2_lateral_conv = self.p2_lateral_conv(self.backbone_model.get_layer('conv2_block3_out').output)
        p2_lateral_conv = tf.reshape(p2_lateral_conv, [batch_size, 160, 160, 256])
        p3_upsampled = UpSampling2D(size=(2, 2), name='p3_upsampled')(p3)
        p2_add = Add(name='p2_add')([p2_lateral_conv, p3_upsampled])
        p2 = Conv2D(256, kernel_size=(3, 3), strides=1, padding='same', activation=None, name='p2_conv_out')(p2_add)
        
        features = [p2, p3, p4, p5, p6]

        # K=2 since face or not face
        classification_outputs = []
        for feature_layer in features:
            classification_outputs.append(ClassificationSubnet(K=1, A=3, prior=.01)(feature_layer))
        classification_outputs = tf.keras.layers.Concatenate(axis=1, name='classification_outputs')(classification_outputs)
        
        # bounding box regression
        regression_outputs = []
        for feature_layer in features:
            regression_outputs.append(RegressionSubnet(n_landmarks=4, A=3)(feature_layer))
        regression_outputs = tf.keras.layers.Concatenate(axis=1, name='regression_outputs')(regression_outputs)
        
#         # facial landmark regression
#         landmarks_outputs = []
#         for feature_layer in features:
#             landmarks_outputs.append(RegressionSubnet(n_landmarks=5, A=3)(feature_layer))
        #return features    
        #return classification_outputs, regression_outputs
        
        outputs = (classification_outputs, regression_outputs)
        
        return outputs
    
    def train_step(self, data):
        # already preprocessed img and bounding boxes
        img, bboxes = data
        #run through RetinaFace model to get regression transformations (n_anchors x 4) and classification scores (n_anchors x K classes)
        outputs = self(img)
        return outputs
        
    def predict_step(self, img):
        pass
#         img = pred_preprocess(img)
#         outputs = self(img)
        
#         outputs = (classification_outputs, regression_outputs)
        
#         # get anchor boxes (based off feature size / stride)
#         sizes = [16, 32, 64, 128, 256]
#         strides = [4, 8, 16, 32, 64]

#         anchors = [Anchors(size=sizes[i], stride=strides[i], name=f'p{i+2}_anchors')(features[i]) for i, f in enumerate(features)]
# #         for x in anchors:
# #             print(x.shape)
#         anchors = tf.keras.layers.Concatenate(axis=1)(anchors) 
#         #return all_anchors
        
#         # apply predicted regression to anchors
#         boxes = RegressBoxes(name='boxes')([anchors, regression_outputs])
#         #make sure they dont go outside image boundary
#         boxes = ClipBoxes(name='clipped_boxes')([inputs, boxes])
        
#         # take boxes 
#         detections = FilterDetections(nms                   = True,
#                                       class_specific_filter = True,
#                                       name                  = 'filtered_detections',
#                                       nms_threshold         = 0.5,
#                                       score_threshold       = 0.05,
#                                       max_detections        = 300,
#                                       parallel_iterations   = 4)([boxes, classification_outputs])
#         return None
    