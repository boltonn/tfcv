import numpy as np
import math

import tensorflow as tf


class AnchorUtils():
    """Create Base Anchor Boxes for Object Detection"""
    
    def __init__(self, resolution=None, sizes=None, strides=None, ratios=None, scales=None, *args, **kwargs):
        """Initiliaze parameters for Anchor Boxes"""
        # strides and sizes align with FPN feature outputs (p2-pn)
        if not resolution:
            resolution = 640
        if not sizes:
            sizes = [16, 32, 64, 128, 256]
        if not strides:
            strides = [4, 8, 16, 32, 64]
            
        # ratios and scales applied to all feature levels from FPN output
        if not ratios:
            #self.ratios  = [0.5, 1, 2]
            ratios = [1] #used in RetinaFace since faces are typically square-like
            
        if not scales:
            scales  = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]
            
        self.resolution = resolution
        self.sizes = sizes
        self.strides = strides
        self.ratios = ratios
        self.scales = scales
        self.n_anchors = len(ratios) * len(scales)
            
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
    
    def shift_and_duplicate(self, anchors, feature_size, stride):
        """Generate bounding boxes by duplicating FPN base anchors every s strides"""
        #feature_size = int(np.round(self.resolution/stride))

        # image_size/stride should equal feature_size (so we could write it for either)
        shift_x = (tf.keras.backend.arange(0, feature_size, dtype=tf.float32) + tf.keras.backend.constant(0.5, dtype=tf.float32)) * stride
        shift_y = (tf.keras.backend.arange(0, feature_size, dtype=tf.float32) + tf.keras.backend.constant(0.5, dtype=tf.float32)) * stride

        # a tensor that supports cartesian indexing
        shift_x, tf.meshgrid(shift_x)
        shift_y = tf.meshgrid(shift_y)

        # make sure of 1-dimensionsal 
        shift_x = tf.keras.backend.reshape(shift_x, [-1])
        shift_y = tf.keras.backend.reshape(shift_y, [-1])
        
        shifts = tf.keras.backend.stack([shift_x, shift_y, shift_x, shift_y], axis=0)
        shifts = tf.keras.backend.transpose(shifts)
        
        k = tf.shape(shifts)[0]
        
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
        
        #generate all anchors for each level of the FPN
        all_anchors = [self.generate_feature_level_base_anchors(size=size) for size in self.sizes]
        all_anchors = [self.shift_and_duplicate(layer_anchors, feature_size, stride) for layer_anchors, feature_size, stride in zip(all_anchors, self.feature_sizes, self.strides)]
        all_anchors = tf.concat(all_anchors, axis=0)

        return all_anchors
    
    

class Anchors(tf.keras.layers.Layer):
    """Create anchor boxes for Object Detection"""
    
    #meant tp run on feature so init is slightly different
    def __init__(self, size, stride, ratios=None, scales=None, *args, **kwargs):
        """Initiliaze parameters for Anchor Boxes"""
        super(Anchors, self).__init__()
        # strides and sizes align with FPN feature outputs (p2-pn)
        self.size = size
        self.stride = stride
        # ratios and scales applied to all feature levels from FPN output
        if not ratios:
            ratios = [1] #used in RetinaFace since faces are typically square-like
            #ratios  = [0.5, 1, 2]
        self.ratios = ratios
        
        if not scales:
            scales  = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]
        self.scales = scales
        self.n_anchors = len(ratios) * len(scales)
        self.anchor_utils = AnchorUtils(ratios=self.ratios, scales=self.scales)
    
    def compute_output_shape(self, input_shape):
        if None not in input_shape[1:]:
            total = np.prod(input_shape[1:3]) * self.n_anchors
            return (input_shape[0], total, 4)
        else:
            return (input_shape[0], None, 4)
    
    def call(self, inputs, **kwargs):
        features = inputs
        features_shape = tf.shape(features)
        batch_size = features_shape[0]
        feature_size = features_shape[1] #reserve first dimension for batch size
        
        anchors = self.anchor_utils.generate_feature_level_base_anchors(self.size)
        feature_level_anchors = self.anchor_utils.shift_and_duplicate(anchors=anchors, feature_size=feature_size, stride=self.stride)
        #duplicate for each image in the batch
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


def tf_compute_iou_map_fn(bbox, anchors):
    """Compute IoU for each annotation/bbox and anchor box combination
        Most other repos will process data as either numpy arrays, sometimes converted to cython code for extra speed.
        However, Tensorflow Datasets is built to load data as TF Records so it generally makes sense to keep everything as Tensors.
            * TF Object detection API: (https://github.com/tensorflow/models/blob/master/research/object_detection/utils/np_box_list_ops.py#L90)
            * Keras RetinaNet Tf 2.0: (https://github.com/fizyr/keras-retinanet/blob/master/keras_retinanet/utils/compute_overlap.pyx); cython code
          
          I have created this function to get decent run times by avoiding for loops but keep everything in Tensorflow.
          We can do this because the anchors stay the same for every image so we can parallelize all bbox calulations against those anchors but it still feels like a hack.

    
    Args:
        bbox (Tensor): (N, 4) tensor of all annotated bounding boxes of form (xmin, ymin, xmax, ymax) unnormalized
        anchors (Tensor): (K, 4) tensor of all anchor boxes (unnormalized)
        
    Returns:
        iou (Tensor): (N, K) tensor of overlap values between 0 and 1
    """    
    anchors = tf.cast(anchors, dtype=tf.float32)
    bbox = tf.cast(bbox, dtype=tf.float32)
    
    #taking advantage of TF's broadcasting
    intersection_xmin = tf.math.maximum(anchors[:, 0], bbox[0])
    intersection_xmax = tf.math.minimum(anchors[:, 2], bbox[2])
    intersection_ymin = tf.math.maximum(anchors[:, 1], bbox[1])
    intersection_ymax = tf.math.minimum(anchors[:, 3], bbox[3])
    intersection = tf.math.maximum(0., (intersection_xmax - intersection_xmin)) * tf.math.maximum(0., (intersection_ymax - intersection_ymin))
    anchor_areas = (anchors[:, 2] - anchors[:, 0]) * (anchors[:, 3] - anchors[:, 1])
    
    bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    union = anchor_areas + bbox_area - intersection
    iou = intersection/union
    return iou


def tf_compute_gt_indices(
    anchors,
    bboxes,
    negative_iou_thresh=0.3,
    positive_iou_thresh=0.5
):
    """ Obtain indices of gt annotations with the greatest overlap.
    
    Args:
        anchors (tensor): np.array of annotations of shape (N, 4) for (x1, y1, x2, y2).
        bboxes (tensor): np.array of shape (N, 5) for (x1, y1, x2, y2, label).
        negative_iou_thresh (float): IoU overlap for negative anchors (all anchors with overlap < negative_iou_thresh are negative).
            * RetinaNet uses 0.4 but RetinaFace used 0.3
        positive_iou_thresh (float): IoU overlap or positive anchors (all anchors with overlap > positive_iou_thresh are positive).
        
    Returns:
        positive_indices (tensor): Tensor of anchor indices that contain an object (>= positive_iou_thresh) [N x 1]
        ignore_indices (tensor): indices of ignored anchor [N x 1]
        negative_indices (tensor): Tensor of anchor indices that are background (< negative_iou_thresh) [N x 1]
        max_iou_indices (tensor): ordered indices of anchors with max IoU [N x 1]
    """
    
    ious =  tf.map_fn(fn=lambda x: tf_compute_iou_map_fn(x, anchors), elems=bboxes)
    print(f'bboxes: {bboxes.get_shape()}')
    print(f'anchors: {anchors.get_shape()}')
    ious = tf.transpose(ious)
    print(f'ious: {ious.get_shape()}\n')
    # indices of the bbox annotation that each anchor box overlaps most with
    max_iou_indices = tf.math.argmax(ious, axis=1)
    # the IoU's for those indices
    max_ious = tf.math.reduce_max(ious, axis=1) 
    
    positive_indices = tf.where(tf.math.greater_equal(max_ious, positive_iou_thresh))
    
    ignore_indices = tf.where(tf.math.logical_and(tf.math.greater_equal(max_ious, negative_iou_thresh), tf.math.less(max_ious, positive_iou_thresh)))
    
    negative_indices = tf.where(tf.math.less(max_ious, negative_iou_thresh))

    return positive_indices, ignore_indices, negative_indices, max_iou_indices


def compute_gt_transforms(anchors, gt_bboxes, mean=0.0, std=0.2):
    """Compute ground-truth transformations from anchor boxes and corresponding g.t. bounding boxes
    
    Args:
        anchors (tensor): anchor boxes unnormalized (N x 4)
        gt_bboxes (tensor): bounding boxes with the highest IoU for each anchor g. t. bboxes (N x 4)
    
    Returns:
        targets (tensor): target transforms [batch_size x ]
    
    """
    anchor_widths  = anchors[:, 2] - anchors[:, 0]
    anchor_heights = anchors[:, 3] - anchors[:, 1]

    # According to the information provided by a keras-retinanet author, they got marginally better results using
    # the following way of bounding box parametrization.
    # See https://github.com/fizyr/keras-retinanet/issues/1273#issuecomment-585828825 for more details
    targets_dx1 = (gt_bboxes[:, 0] - anchors[:, 0]) / anchor_widths
    targets_dy1 = (gt_bboxes[:, 1] - anchors[:, 1]) / anchor_heights
    targets_dx2 = (gt_bboxes[:, 2] - anchors[:, 2]) / anchor_widths
    targets_dy2 = (gt_bboxes[:, 3] - anchors[:, 3]) / anchor_heights
    
    targets = tf.stack((targets_dx1, targets_dy1, targets_dx2, targets_dy2), axis=1)

    targets = (targets - mean) / std
    return targets



def compute_targets(anchors, bboxes, num_classes, labels=None, negative_iou_thresh=0.3, positive_iou_thresh=0.5):
    """Compute Classification and Regression Targets for Anchor Box dependent loss
    
    Args:
        anchors (tensor): anchor boxes of shape [N x 4]
        bboxes (tensor): unnormalized bounding boxes if format (x1, x2, y1, y2) of shape [K x 4]
        num_classes (int): Number of classes
        labels (tensor): Tensor of tf.int for each bbox if more than one class; else assume binary [K x 1]
        negative_iou_thresh (float): All anchors < negative_iou_thresh are consider background or 'negative' examples (anchor state 0)
        positive_iou_thresh (float): All anchors >= positive_iou_thresh are used as 'positive' examples for the loss (anchor state 1)
            * anything in between is set to 'ignore' (anchor state -1)
    
    Returns:
        classification_targets (tensor): Tensor of one-hot encoded labels for all bboxes with highest IoU per anchor, plus anchor state column (N, num_classes + 1)
            * anchor state column -> (1) for positive anchor boxes, (0) for negative, (-1) for those to be ignored
        regression_targets (tensor): Tensor of ground truth transformations applied to positive anchor boxes to get ground truth bounding boxes (N, 4 + 1)
            * 4 + 1 = 4 transformations on each coordinate + same anchor state column as classification targets
    """
    positive_indices, ignore_indices, negative_indices, max_iou_indices = tf_compute_gt_indices(anchors, bboxes, negative_iou_thresh=0.4, positive_iou_thresh=0.5)
    
    #create the sine column for whether a anchor is background (0), an object (1), or should be ignore (-1)
    iou_sine_col = tf.zeros(anchors.get_shape()[0])
    pos_iou_sine_col = tf.zeros(anchors.get_shape()[0])
    if positive_indices.get_shape()[0]!=0:
        # we call this something else b/c we can use it to get the positive classes matrix
        pos_iou_sine_col = tf.tensor_scatter_nd_add(iou_sine_col, positive_indices, tf.ones(tf.shape(positive_indices)[0]))
    if ignore_indices.get_shape()[0]!=0:
        iou_sine_col = tf.tensor_scatter_nd_sub(pos_iou_sine_col, ignore_indices, tf.ones(tf.shape(ignore_indices)[0]))
        
    #create the class targets (N, K+1)
    def _map_class(max_iou_indices, labels):
        """Fast way to map indexes of boxes to corresponsing labels"""
        #add on index column
        max_iou_indices = tf.stack([tf.reshape(tf.convert_to_tensor([np.arange(0, tf.shape(all_anchors)[0])]), [1, tf.shape(max_iou_indices)[0]]),
                                    tf.cast(tf.expand_dims(max_iou_indices, axis=0), dtype=tf.int32)], axis=0)
        max_iou_indices = tf.transpose(tf.squeeze(max_iou_indices))
        broadcasted_labels = tf.broadcast_to(labels, [tf.shape(all_anchors)[0], tf.shape(random_labels)[0]])
        anchor_classes = tf.gather_nd(broadcasted_labels, temp)
        return anchor_classes

    if num_classes<=2:
        classification_targets = tf.transpose(tf.stack([pos_iou_sine_col, iou_sine_col], axis=0))
    else:
        assert labels is not None, "Labels as tensor of ints for each bbox need to be passed if multiple classes"
        # map the bbox index that each anchor overlaps with the most to the corresponsing label
        # this is very slow so need to come back to find a better way
        anchor_classes = _map_class(max_iou_indices, labels)
        
        # keep only the positive ones (swap with -1 since tensorflow make -1 become 0 and one-hot enconding)
        anchor_classes = tf.tensor_scatter_nd_update(anchor_classes, ignore_indices, tf.constant(-1, shape=tf.shape(ignore_indices)[0], dtype=tf.int32))
        anchor_classes = tf.tensor_scatter_nd_update(anchor_classes, negative_indices, tf.constant(-1, shape=tf.shape(negative_indices)[0], dtype=tf.int32))

        class_matrix = tf.one_hot(tf.cast(anchor_classes, tf.int32), num_classes)
        
        #add on the sine col
        classification_targets = tf.concat([class_matrix, tf.expand_dims(iou_sine_col, -1)], axis=1)
    
    #create regression targets (N, 4 + 1)
    #closest bounding box to each anchor
    gt_bboxes = tf.gather(bboxes, max_iou_indices) # (N, 4)
    
    regression_matrix = compute_gt_transforms(anchors, gt_bboxes, mean=0.0, std=0.2)
    #add on the sine col
    regression_targets = tf.concat([regression_matrix, tf.expand_dims(iou_sine_col, -1)], axis=1)
    return (classification_targets, regression_targets)


def filter_anchors(anchors, classification_targets, regression_targets, img_width=640, img_height=640):
    """Filter anchor boxes (set anchor state to 'ignore' (-1) for classification and regression targets ) whose center isn't the image
    
    Args:
        anchors (tensor): unnormalized anchor boxes (x1, y1, x2, y2) of shape [N x 4]
        classification_targets (tensor): Tensor of one-hot encoded labels for all bboxes with highest IoU per anchor, plus anchor state column (N, num_classes + 1)
            * anchor state column -> (1) for positive anchor boxes, (0) for negative, (-1) for those to be ignored
        regression_targets (tensor): Tensor of ground truth transformations applied to positive anchor boxes to get ground truth bounding boxes (N, 4 + 1)
            * 4 + 1 = 4 transformations on each coordinate + same anchor state column as classification targets
        img_width (int): image width
        img_height (int): image height
        
    Returns:
        classification_targets (tensor): same tensor but with anchor state 'ignore' / (-1) for any anchor box whose center isnt in the image (N, num_classes + 1)
        regression_targets (tensor): same tensor but with anchor state 'ignore' / (-1) for any anchor box whose center isnt in the image (N, 4 + 1) 
    """
    anchor_centers = tf.transpose(tf.stack([(anchors[:, 0] + anchors[:, 2]) / 2, (anchors[:, 1] + anchors[:, 3]) / 2]))
    
    outside_wdith_indices = tf.math.logical_or(tf.math.greater_equal(anchor_centers[:, 0], img_width), tf.math.less_equal(anchor_centers[:, 0], 0.))
    outside_height_indices = tf.math.logical_or(tf.math.greater_equal(anchor_centers[:, 0], img_height), tf.math.less_equal(anchor_centers[:, 0], 0.))
    ignore_indices = tf.math.logical_or(outside_wdith_indices, outside_height_indices)
    
    #update
    if tf.shape(ignore_indices)[0]!=0:
        classification_targets = tf.tensor_scatter_nd_update(classification_targets, ignore_indices, tf.constant(-1, shape=tf.shape(ignore_indices)[0], dtype=tf.float32))
        regression_targets = tf.tensor_scatter_nd_update(regression_targets, ignore_indices, tf.constant(-1, shape=tf.shape(ignore_indices)[0], dtype=tf.float32))

    return (classification_targets, regression_targets)
    
    
# Inferene Utils
    
def compute_pred_boxes(deltas, anchors, mean=0.0, std=0.2):
    """Get Predicted Boxes from predicted deltas and anchors"""
    #first dimension is the batch size
    width  = anchors[:, :, 2] - anchors[:, :, 0]
    height = anchors[:, :, 3] - anchors[:, :, 1]

    x1 = anchors[:, :, 0] + (deltas[:, :, 0] * std[0] + mean[0]) * width
    y1 = anchors[:, :, 1] + (deltas[:, :, 1] * std[1] + mean[1]) * height
    x2 = anchors[:, :, 2] + (deltas[:, :, 2] * std[2] + mean[2]) * width
    y2 = anchors[:, :, 3] + (deltas[:, :, 3] * std[3] + mean[3]) * height

    pred_boxes = tf.stack([x1, y1, x2, y2], axis=2)

    return pred_boxes