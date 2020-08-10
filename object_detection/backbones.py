def load_backbone(name='resnet50', input_shape=(640, 640, 3)):
    if name=='resnet50':
        model =  tf.keras.applications.ResNet50(weights='imagenet', input_shape=(640, 640, 3), include_top=False)

        backbone_model = tf.keras.Model(inputs=model.input,
                                        outputs=[model.get_layer("conv4_block6_out").output,
                                                 model.get_layer("conv3_block4_out").output,
                                                 model.get_layer("conv2_block3_out").output,
                                                 model.output])
    else:
        raise ValueError("Backbone model must be one of 'resnet50' or 'resnet152'")
    return backbone_model