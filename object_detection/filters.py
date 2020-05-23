import numpy as np
import tensorflow as tf


def compute_gt_delta(anchors, gt_boxes, mean=0.0, std=0.2):
    """Compute ground-truth transformations from annotations and anchor boxes"""
    anchor_widths  = anchors[:, 2] - anchors[:, 0]
    anchor_heights = anchors[:, 3] - anchors[:, 1]

    # According to the information provided by a keras-retinanet author, they got marginally better results using
    # the following way of bounding box parametrization.
    # See https://github.com/fizyr/keras-retinanet/issues/1273#issuecomment-585828825 for more details
    targets_dx1 = (gt_boxes[:, 0] - anchors[:, 0]) / anchor_widths
    targets_dy1 = (gt_boxes[:, 1] - anchors[:, 1]) / anchor_heights
    targets_dx2 = (gt_boxes[:, 2] - anchors[:, 2]) / anchor_widths
    targets_dy2 = (gt_boxes[:, 3] - anchors[:, 3]) / anchor_heights
    
    targets = tf.concat((targets_dx1, targets_dy1, targets_dx2, targets_dy2), axis=0)
    targets = targets.T

    targets = (targets - mean) / std
    return targets


def compute_pred_boxes(anchors, deltas, mean=0.0, std=0.2):
    """Get Predicted Boxes from predicted deltas and anchors"""
    #first dimension is the batch size
    width  = anchors[:, :, 2] - anchors[:, :, 0]
    height = anchors[:, :, 3] - anchors[:, :, 1]

    x1 = anchors[:, :, 0] + (deltas[:, :, 0] * std[0] + mean[0]) * width
    y1 = anchors[:, :, 1] + (deltas[:, :, 1] * std[1] + mean[1]) * height
    x2 = anchors[:, :, 2] + (deltas[:, :, 2] * std[2] + mean[2]) * width
    y2 = anchors[:, :, 3] + (deltas[:, :, 3] * std[3] + mean[3]) * height

    pred_boxes = tf.keras.backend.stack([x1, y1, x2, y2], axis=2)

    return pred_boxes


class RegressBoxes(tf.keras.layers.Layer):
    """ Keras layer for applying regression values to boxes.
    """

    def __init__(self, mean=None, std=None, *args, **kwargs):
        """ Initializer for the RegressBoxes layer.
        Args
            mean: The mean value of the regression values which was used for normalization.
            std: The standard value of the regression values which was used for normalization.
        """
        if mean is None:
            mean = np.array([0, 0, 0, 0])
        if std is None:
            std = np.array([0.2, 0.2, 0.2, 0.2])

        if isinstance(mean, (list, tuple)):
            mean = np.array(mean)
        elif not isinstance(mean, np.ndarray):
            raise ValueError('Expected mean to be a np.ndarray, list or tuple. Received: {}'.format(type(mean)))

        if isinstance(std, (list, tuple)):
            std = np.array(std)
        elif not isinstance(std, np.ndarray):
            raise ValueError('Expected std to be a np.ndarray, list or tuple. Received: {}'.format(type(std)))

        self.mean = mean
        self.std  = std
        super(RegressBoxes, self).__init__(*args, **kwargs)

    def call(self, inputs, **kwargs):
        anchors, regression_outputs = inputs
        return compute_pred_boxes(anchors, regression_outputs, mean=self.mean, std=self.std)

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        config = super(RegressBoxes, self).get_config()
        config.update({
            'mean': self.mean.tolist(),
            'std' : self.std.tolist(),
        })

        return config
    
    
class ClipBoxes(tf.keras.layers.Layer):
    """ Keras layer to clip box values to lie inside a given shape.
    """
    def call(self, inputs, **kwargs):
        image, boxes = inputs
        image_shape = tf.cast(tf.shape(image), tf.keras.backend.floatx())
        _, width, height, _ = tf.unstack(image_shape, axis=0)

        x1, y1, x2, y2 = tf.unstack(boxes, axis=-1)
        x1 = tf.clip_by_value(x1, 0, width  - 1)
        y1 = tf.clip_by_value(y1, 0, height - 1)
        x2 = tf.clip_by_value(x2, 0, width  - 1)
        y2 = tf.clip_by_value(y2, 0, height - 1)

        return tf.stack([x1, y1, x2, y2], axis=2)

    def compute_output_shape(self, input_shape):
        return input_shape[1]
    
    
    
def filter_detections(
    boxes,
    classification,
    other                 = [],
    class_specific_filter = True,
    nms                   = True,
    score_threshold       = 0.05,
    max_detections        = 300,
    nms_threshold         = 0.5
):
    """ Filter detections using the boxes and classification values.
    Args
        boxes                 : Tensor of shape (num_boxes, 4) containing the boxes in (x1, y1, x2, y2) format.
        classification        : Tensor of shape (num_boxes, num_classes) containing the classification scores.
        other                 : List of tensors of shape (num_boxes, ...) to filter along with the boxes and classification scores.
        class_specific_filter : Whether to perform filtering per class, or take the best scoring class and filter those.
        nms                   : Flag to enable/disable non maximum suppression.
        score_threshold       : Threshold used to prefilter the boxes with.
        max_detections        : Maximum number of detections to keep.
        nms_threshold         : Threshold for the IoU value to determine when a box should be suppressed.
    Returns
        A list of [boxes, scores, labels, other[0], other[1], ...].
        boxes is shaped (max_detections, 4) and contains the (x1, y1, x2, y2) of the non-suppressed boxes.
        scores is shaped (max_detections,) and contains the scores of the predicted class.
        labels is shaped (max_detections,) and contains the predicted label.
        other[i] is shaped (max_detections, ...) and contains the filtered other[i] data.
        In case there are less than max_detections detections, the tensors are padded with -1's.
    """
    def _filter_detections(scores, labels):
        # threshold based on score
        indices = tf.where(tf.math.greater(scores, score_threshold))

        if nms:
            filtered_boxes  = tf.gather_nd(boxes, indices)
            filtered_scores = tf.gather(scores, indices)[:, 0]

            # perform NMS
            nms_indices = tf.image.non_max_suppression(filtered_boxes, filtered_scores, max_output_size=max_detections, iou_threshold=nms_threshold)

            # filter indices based on NMS
            indices = tf.gather(indices, nms_indices)

        # add indices to list of all indices
        labels = tf.gather_nd(labels, indices)
        indices = tf.stack([indices[:, 0], labels], axis=1)

        return indices

    if class_specific_filter:
        all_indices = []
        # perform per class filtering
        for c in range(int(classification.shape[1])):
            scores = classification[:, c]
            labels = c * tf.ones((tf.shape(scores)[0],), dtype='int64')
            all_indices.append(_filter_detections(scores, labels))

        # concatenate indices to single tensor
        indices = tf.concat(all_indices, axis=0)
    else:
        scores  = tf.keras.backend.max(classification, axis    = 1)
        labels  = tf.math.argmax(classification, axis = 1)
        indices = _filter_detections(scores, labels)

    # select top k
    scores              = tf.gather_nd(classification, indices)
    labels              = indices[:, 1]
    scores, top_indices = tf.math.top_k(scores, k=tf.math.minimum(max_detections, tf.shape(scores)[0]))

    # filter input using the final set of indices
    indices             = tf.gather(indices[:, 0], top_indices)
    boxes               = tf.gather(boxes, indices)
    labels              = tf.gather(labels, top_indices)
    other_              = [tf.gather(o, indices) for o in other]

    # zero pad the outputs
    pad_size = tf.math.maximum(0, max_detections - tf.shape(scores)[0])
    boxes    = tf.pad(boxes, [[0, pad_size], [0, 0]], constant_values=-1)
    scores   = tf.pad(scores, [[0, pad_size]], constant_values=-1)
    labels   = tf.pad(labels, [[0, pad_size]], constant_values=-1)
    labels   = tf.cast(labels, 'int32')
    other_   = [tf.pad(o, [[0, pad_size]] + [[0, 0] for _ in range(1, len(o.shape))], constant_values=-1) for o in other_]

    # set shapes, since we know what they are
    boxes.set_shape([max_detections, 4])
    scores.set_shape([max_detections])
    labels.set_shape([max_detections])
    for o, s in zip(other_, [list(keras.backend.int_shape(o)) for o in other]):
        o.set_shape([max_detections] + s[1:])

    return [boxes, scores, labels] + other_


class FilterDetections(tf.keras.layers.Layer):
    """ Keras layer for filtering detections using score threshold and NMS.
    """

    def __init__(
        self,
        nms                   = True,
        class_specific_filter = True,
        nms_threshold         = 0.5,
        score_threshold       = 0.05,
        max_detections        = 300,
        parallel_iterations   = 32,
        **kwargs
    ):
        """ Filters detections using score threshold, NMS and selecting the top-k detections.
        Args
            nms                   : Flag to enable/disable NMS.
            class_specific_filter : Whether to perform filtering per class, or take the best scoring class and filter those.
            nms_threshold         : Threshold for the IoU value to determine when a box should be suppressed.
            score_threshold       : Threshold used to prefilter the boxes with.
            max_detections        : Maximum number of detections to keep.
            parallel_iterations   : Number of batch items to process in parallel.
        """
        self.nms                   = nms
        self.class_specific_filter = class_specific_filter
        self.nms_threshold         = nms_threshold
        self.score_threshold       = score_threshold
        self.max_detections        = max_detections
        self.parallel_iterations   = parallel_iterations
        super(FilterDetections, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        """ Constructs the NMS graph.
        Args
            inputs : List of [boxes, classification, other[0], other[1], ...] tensors.
        """
        boxes          = inputs[0]
        classification = inputs[1]
        other          = inputs[2:]

        # wrap nms with our parameters
        #@tf.function
        def _filter_detections(args):
            boxes          = args[0]
            classification = args[1]
            other          = args[2]

            return filter_detections(
                boxes,
                classification,
                other,
                nms                   = self.nms,
                class_specific_filter = self.class_specific_filter,
                score_threshold       = self.score_threshold,
                max_detections        = self.max_detections,
                nms_threshold         = self.nms_threshold,
            )

        # call filter_detections on each batch
        outputs = tf.map_fn(
            _filter_detections,
            elems=[boxes, classification, other],
            dtype=[tf.keras.backend.floatx(), tf.keras.backend.floatx(), tf.int32] + [o.dtype for o in other],
            parallel_iterations=self.parallel_iterations
        )

        return outputs

    def compute_output_shape(self, input_shape):
        """ Computes the output shapes given the input shapes.
        Args
            input_shape : List of input shapes [boxes, classification, other[0], other[1], ...].
        Returns
            List of tuples representing the output shapes:
            [filtered_boxes.shape, filtered_scores.shape, filtered_labels.shape, filtered_other[0].shape, filtered_other[1].shape, ...]
        """
        return [
            (input_shape[0][0], self.max_detections, 4),
            (input_shape[1][0], self.max_detections),
            (input_shape[1][0], self.max_detections),
        ] + [
            tuple([input_shape[i][0], self.max_detections] + list(input_shape[i][2:])) for i in range(2, len(input_shape))
        ]

    def compute_mask(self, inputs, mask=None):
        """ This is required in Keras when there is more than 1 output.
        """
        return (len(inputs) + 1) * [None]

    def get_config(self):
        """ Gets the configuration of this layer.
        Returns
            Dictionary containing the parameters of this layer.
        """
        config = super(FilterDetections, self).get_config()
        config.update({
            'nms'                   : self.nms,
            'class_specific_filter' : self.class_specific_filter,
            'nms_threshold'         : self.nms_threshold,
            'score_threshold'       : self.score_threshold,
            'max_detections'        : self.max_detections,
            'parallel_iterations'   : self.parallel_iterations,
        })

        return config