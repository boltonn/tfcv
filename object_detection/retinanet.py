class RetinaNet(tf.keras.Model):
    """A subclassed Keras model implementing the RetinaNet architecture.

    Attributes:
      num_classes: Number of classes in the dataset.
      backbone: The backbone to build the feature pyramid from.
        Currently supports ResNet50 only.
    """

    def __init__(self, n_classes, backbone=None, **kwargs):
        super(RetinaNet, self).__init__(name="RetinaNet", **kwargs)
        self.num_classes = num_classes
        self.fpn = FeaturePyramid(backbone)
        self.classification_subnet = ClassificationSubnet(n_classes, n_base_anchors=9, prior=0.01)
        self.regression_subnet = RegressionSubnet(n_landmarks=4, n_base_anchors=9)

    def call(self, image, training=False):
        features = self.fpn(image, training=training)
        cls_outputs = []
        box_outputs = []
        for feature in features:
            box_outputs.append(self.regression_subnet(feature))
            cls_outputs.append(self.classification_subnet(feature))
        cls_outputs = tf.concat(cls_outputs, axis=1)
        box_outputs = tf.concat(box_outputs, axis=1)
        return tf.concat([box_outputs, cls_outputs], axis=-1)
