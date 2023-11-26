import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, UpSampling2D, add, Conv2DTranspose, Dropout, Activation, SeparableConv2D, ReLU
from tensorflow.keras import Model, Input


input_shapes = {'darea_size': [80, 160, 3], 
                'mobnet_shape': [224, 224, 3],
                'mobnet_shape2' : [128, 128, 3],
                'model_in_size2' : [96, 192, 3]}

# u-Net like 2 Model from Keras Hub
def u_netlike_model2(img_size, num_classes, dropout_rate=0.2):
    inputs = Input(shape=img_size)

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128]:
        x = Activation("relu")(x)
        x = SeparableConv2D(filters, 3, padding="same")(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)

        x = Activation("relu")(x)
        x = SeparableConv2D(filters, 3, padding="same")(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)

    x = MaxPooling2D(3, strides=2, padding="same")(x)
    # Project residual
    residual = Conv2D(filters, 1, strides=2, padding="same")(previous_block_activation)
    print(residual.shape)
    print(x.shape)

    x = add([x, residual])  # Add back residual
    previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###
    for filters in [128, 64, 32]:
        x = Activation("relu")(x)

        x = Conv2DTranspose(filters, 3, padding="same")(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        x = Activation("relu")(x)

        x = Conv2DTranspose(filters, 3, padding="same")(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        
    x = UpSampling2D(2)(x)
    # Project residual
    residual = UpSampling2D(2)(previous_block_activation)
    residual = Conv2D(filters, 1, padding="same")(residual)

    print(residual.shape)
    print(x.shape)
    x = add([x, residual])  # Add back residual
    
    previous_block_activation = x  # Set aside next residual
    
    # Add a per-pixel classification layer
    x = UpSampling2D(interpolation='bilinear')(x)
    x = Conv2D(num_classes, 16, activation="softmax", padding="same")(x)
    x = Dropout(dropout_rate)(x)
    outputs = Conv2D(num_classes, 3, activation="softmax", padding="same")(x)
    
    # Define the model
    model = tf.keras.Model(inputs, outputs)

    return model

# model3 = u_netlike_model2(model_in_size, 3, dropout_rate=0.2) 
# model3.summary(line_length =100)
# tf.keras.utils.plot_model(model3, "u_net_like_model.png")

# U-Net like model defnition User defined model
def u_net_like_model(dropout_rate=0.5):
    input_x = Input(shape=model_in_size)
    
    # Encoder-1
    x0 = Conv2D(64, (3,3), padding='valid', strides=(1,1), activation = 'relu')(input_x)
    x = Conv2D(64, (3,3), padding='valid', strides=(1,1), activation = 'relu')(x0)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2,2))(x)

    # Encoder-2
    x1 = Conv2D(128, (3,3), padding='valid', strides=(1,1), activation = 'relu')(x)
    x = BatchNormalization()(x1)
    x2 = MaxPooling2D(pool_size=(2,2))(x)

    # Encoder-3
    x3 = Conv2D(256, (3,3), padding='valid', strides=(1,1), activation = 'relu')(x2)
    x = BatchNormalization()(x3)
    x = MaxPooling2D(pool_size=(2,2))(x)

    # Decoder-1
    y1 = UpSampling2D(interpolation='bilinear')(x)
    y = add([y1, x3])

    # Decoder-2
    y = Conv2DTranspose(128, (3,3), padding='valid', strides=(1,1), activation = 'relu')(y)
    y2 = UpSampling2D(interpolation='bilinear')(y)
    y = add([y2, x1])

    # Decoder-3
    y = Conv2DTranspose(64, (3,3), padding='valid', strides=(1,1), activation = 'relu')(y)
    y = UpSampling2D(interpolation='bilinear')(y)
    y3 = Conv2DTranspose(64, (3,3), padding='valid', strides=(1,1), activation = 'relu')(y)
    y = add([y3, x0])

    # final size adjustment
    y = Conv2DTranspose(32, (3,3), padding='valid', strides=(1,1), activation = 'relu')(y)
    y = Conv2D(16, (1,1), padding='valid', strides=(1,1), activation ="softmax")(y)
    y = Conv2D(8, (1,1), padding='valid', strides=(1,1), activation ="softmax")(y)
    y4 = Conv2D(3, (1,1), padding='valid', strides=(1,1), activation ="softmax")(y)
    
    return Model(input_x, y4)

# model2 = u_net_like_model(dropout_rate=0.2) 
# model2.summary(line_length =100)
# tf.keras.utils.plot_model(model, "u_net_like_model.png")

# The simple model
def simple_encoder_decoder():
    """
    Try a simple Encoder-Decoder architecture.
    Use Convolutions, Transposed Convolutions.
    Use Pooling, Unpooling or UpSampling.
    GOAL: Match the output to the input. Experiment.
    """
    ## YOUR CODE HERE
    input_x = Input(shape = model_in_size)
    x = BatchNormalization(input_shape=(80,160,3))(input_x)
    x = Conv2D(32, (3, 3), padding='valid', strides=(1,1), activation = 'relu')(x)
    x1 = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2,2))(x1)
    x = UpSampling2D(interpolation='bilinear')(x)
    x = add([x1, x])
    x = Conv2DTranspose(32,(3,3),strides=(1,1), padding='valid', activation='relu')(x)
    x = Conv2D(3, (1,1), strides=(1,1), padding='valid',activation='softmax')(x)
    return Model(input_x,x)

# model1 = simple_encoder_decoder() 
# model1.summary()
# tf.keras.utils.plot_model(model1, "basic_segmentaion_model.png")


class unet_from_mobilenet:

    # Model definition
    def __init__(self):

        self.input_shapes = input_shapes['mobnet_shape2']

        self.base_model = tf.keras.applications.MobileNetV2(input_shape = input_shapes['mobnet_shape2'], include_top = False)
        # tf.keras.utils.plot_model(base_model, "mobileNetV2_model_80_160.png", show_shapes=True)

        self.layer_names = ['block_1_expand_relu',
                        'block_3_expand_relu',
                        'block_6_expand_relu',
                        'block_13_expand_relu',
                        'block_16_project']

        self.base_model_outputs = [self.base_model.get_layer(name).output for name in self.layer_names]
        self.down_stack

        down_stack = tf.keras.Model(inputs= self.base_model.input, 
                                    outputs = self.base_model_outputs,
                                    name='pretrained_mobilenet')

        down_stack.trainable = True # True givrs better performance

    def upsample (self, filters, size, name):
        return tf.keras.Sequential([
            Conv2DTranspose(filters, size, strides=2, padding='same'),
            BatchNormalization(),
            Dropout(0.5),
            ReLU()
        ])

    up_stack = [upsample(512, 3, 'upsample_4*4_to_8*8'),
                upsample(256, 3, 'upsample_8*8_to_16*16'),
                upsample(128, 3, 'upsample_16*16_to_32*32'),
                upsample(64, 3, 'upsample_32*32_to_64*64')]

    def unet_model3(self, model_in_shape, output_channel):

        inputs = Input(shape=model_in_shape)
        skips = self.down_stack(inputs)
        x = skips[-1]
        skips = reversed(skips[:-1])
        
        # concatenating
        for up, skip in zip(self.up_stack, skips):
            x = up(x)
            concat = tf.keras.layers.Concatenate()
            x = concat([x, skip])

        # last layer of the model
        output = Conv2DTranspose(output_channel, 3, strides=2, padding='same', activation='softmax')(x)
        
        return Model(inputs=inputs, outputs=output)


model = unet_from_mobilenet.unet_model3(input_shapes['mobnet_shape2'], 3)