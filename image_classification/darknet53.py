import tensorflow as tf
import tensorflow_addons as tfa

class ResidualBlock(tf.keras.layers.Layer):
    """Darknet Residual Block"""
    def __init__(self, filter_num, stride=1, activation='mish'):
        super(ResidualBlock, self).__init__()
        self.convblock1 = ConvBlock(filter_num, kernel_size=(1, 1), activation=activation)
        self.convblock2 = ConvBlock(filter_num*2, kernel_size=(3, 3), activation=activation)

    def call(self, inputs):
        x = self.convblock1(inputs)
        x = self.convblock2(x)
        x = tf.keras.layers.add([inputs, x])
        if activation=='mish':
            output = tfa.activations.mish(x)
        else:
            output = tf.nn.leaky_relu(x)
        return output

class ConvBlock(tf.keras.layers.Layer):
    """Darknet Convolutional Block"""
    def __init__(self, filter_num, kernel_size, stride=1, activation='mish'):
        super(ConvBlock, self).__init__()
        self.conv = tf.keras.layers.Conv2D(filters=filter_num,
                                           kernel_size=kernel_size,
                                           strides=stride,
                                           padding="same")
        self.bn = tf.keras.layers.BatchNormalization()
            
    def call(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        if activation=='mish':
            output = tfa.activations.mish(x)
        else:
            output = tf.nn.leaky_relu(x)
        return output

# tf.keras.backend.clear_session() 
class Darknet53(tf.keras.Model):
    def __init__(self, n_classes):
        # 53 is # of convolutions
        super(Darknet53, self).__init__()
        self.convblock1 = ConvBlock(32, kernel_size=(3, 3), stride=1)
        self.convblock2 = ConvBlock(64, kernel_size=(3, 3), stride=2)
        
        self.residual_layer1 = self.make_layer(32, n_blocks=1)
        self.convblock_layer1 = ConvBlock(128, kernel_size=(3, 3), stride=2)
        
        self.residual_layer2 = self.make_layer(64, n_blocks=2)
        self.convblock_layer2 = ConvBlock(256, kernel_size=(3, 3), stride=2)
        
        self.residual_layer3 = self.make_layer(128, n_blocks=8)
        self.convblock_layer3 = ConvBlock(512, kernel_size=(3, 3), stride=2)
        
        self.residual_layer4 = self.make_layer(256, n_blocks=8)
        self.convblock_layer4 = ConvBlock(1024, kernel_size=(3, 3), stride=2)
        
        self.residual_layer5 = self.make_layer(512, n_blocks=8)
        
        self.avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.out = tf.keras.layers.Dense(n_classes, activation='softmax')
        
    def make_layer(self, filter_num, n_blocks, stride=1):
        """Repeat Number of Residual Blocks"""
        residual_blocks=[]
        for _ in range(0, n_blocks):
            residual_blocks.append(ResidualBlock(filter_num))
        return tf.keras.Sequential(residual_blocks)


    def call(self, inputs):
        x = self.convblock1(inputs)
        x = self.convblock2(x)
        x = self.residual_layer1(x)
        x = self.convblock_layer1(x)
        x = self.residual_layer2(x)
        x = self.convblock_layer2(x)
        x = self.residual_layer3(x)
        x = self.convblock_layer3(x)
        x = self.residual_layer4(x)
        x = self.convblock_layer4(x)
        x = self.residual_layer5(x)
        x = self.avg_pool(x)
        output = self.out(x)
        
        return output
    
    
# inputs = tf.keras.Input(shape=(416, 416, 3))
# model = Darknet53(n_classes=1000)

# x = model(inputs, training=False)

