"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import tensorflow as tf

# @tf.function
def focal_loss(y_true, y_pred):
    """Compute Focal Loss"""
    classification_targets = y_true[0]
    regression_targets = y_true[1]
    
    classification_pred = y_pred[0]
    regression_pred = y_pred[1]
        
    classification_loss = _focal(classification_targets, classification_pred)
    regression_loss = _smooth_l1(regression_targets, regression_pred)
    
    return classification_loss + regression_loss



# def focal_loss(y_true, classification_pred, regression_pred):
#     """Compute Focal Loss"""
#     classification_targets = y_true[0]
#     regression_targets = y_true[1]
    
# #     classification_pred = y_pred[0]
# #     regression_pred = y_pred[1]
        
#     classification_loss = _focal(classification_targets, classification_pred)
#     regression_loss = _smooth_l1(regression_targets, regression_pred)
    
#     return classification_loss + regression_loss
    
    


def _focal(y_true, y_pred, cutoff=0.5, alpha=0.25, gamma=2.0):
    """ Compute the focal loss given the target tensor and the predicted tensor.
    
    As defined in https://arxiv.org/abs/1708.02002

    Args:
        y_true: Tensor of target data from the generator with shape (B, N, num_classes)
        y_pred: Tensor of predicted data from the network with shape (B, N, num_classes)
        alpha: Scale the focal weight with alpha
        gamma: Take the power of the focal weight with gamma.

    Returns:
        focal_loss (tensor) The focal loss of y_pred w.r.t. y_true.
    """
    labels         = y_true[:, :, :-1]
    anchor_state   = y_true[:, :, -1]  # -1 for ignore, 0 for background, 1 for object
    print(f'labels: {labels.shape}')
    print(f'anchor_state: {anchor_state.shape}')
    classification = y_pred

    # filter out "ignore" anchors
    indices        = tf.where(tf.math.not_equal(anchor_state, -1))
    labels         = tf.gather_nd(labels, indices)
    classification = tf.gather_nd(classification, indices)

    # compute the focal loss
    alpha_factor = tf.ones_like(labels) * alpha
    alpha_factor = tf.where(tf.math.greater(labels, cutoff), alpha_factor, 1 - alpha_factor)
    focal_weight = tf.where(tf.math.greater(labels, cutoff), 1 - classification, classification)
    focal_weight = alpha_factor * focal_weight ** gamma

    cls_loss = focal_weight * tf.keras.losses.binary_crossentropy(labels, classification)

    # compute the normalizer: the number of positive anchors
    normalizer = tf.where(tf.math.equal(anchor_state, 1))
    normalizer = tf.cast(tf.shape(normalizer)[0], tf.keras.backend.floatx())
    normalizer = tf.math.maximum(tf.keras.backend.cast_to_floatx(1.0), normalizer)

    return tf.keras.backend.sum(cls_loss) / normalizer



def _smooth_l1(y_true, y_pred, sigma=3.0):
    """ Compute the smooth L1 loss of y_pred w.r.t. y_true.
    
    Args:
        y_true: Tensor from the generator of shape (B, N, 5). The last value for each box is the state of the anchor (ignore, negative, positive).
        y_pred: Tensor from the network of shape (B, N, 4).
        sigma: This argument defines the point where the loss changes from L2 to L1.

    Returns:
        The smooth L1 loss of y_pred w.r.t. y_true.
    """
    sigma_squared = sigma ** 2
    # separate target and state
    regression        = y_pred
    regression_target = y_true[:, :, :-1]
    anchor_state      = y_true[:, :, -1]

    # filter out "ignore" and negative anchors
    indices           = tf.where(tf.math.equal(anchor_state, 1))
    regression        = tf.gather_nd(regression, indices)
    regression_target = tf.gather_nd(regression_target, indices)

    # compute smooth L1 loss
    # f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
    #        |x| - 0.5 / sigma / sigma    otherwise
    regression_diff = regression - regression_target
    regression_diff = tf.math.abs(regression_diff)
    regression_loss = tf.where(
        tf.math.less(regression_diff, 1.0 / sigma_squared),
        0.5 * sigma_squared * tf.pow(regression_diff, 2),
        regression_diff - 0.5 / sigma_squared)

    # compute the normalizer: the number of positive anchors
    normalizer = tf.math.maximum(1, tf.shape(indices)[0])
    normalizer = tf.cast(normalizer, dtype=tf.keras.backend.floatx())
    return tf.keras.backend.sum(regression_loss) / normalizer
