import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
import os


def conv2d(inputs, filters, kernel_size, stride=1, non_linear='relu', padding='same', use_bias=False, deepwise=False, batchnorm=True):

    if deepwise:
        x = keras.layers.DepthwiseConv2D(kernel_size=kernel_size, padding=padding, strides=stride,
                                         depthwise_initializer='he_normal',
                                         depthwise_regularizer=keras.regularizers.l2(5e-4),
                                         use_bias=use_bias)(inputs)
    else:
        x = keras.layers.Conv2D(filters=filters, kernel_size=kernel_size,
                                kernel_initializer='he_normal',
                                kernel_regularizer=keras.regularizers.l2(5e-4),
                                strides=stride, padding=padding, use_bias=use_bias)(inputs)

    if batchnorm:
        x = keras.layers.BatchNormalization()(x)

    if non_linear == 'relu':
        x = keras.layers.ReLU()(x)  #ReLU6
    elif non_linear == 'hswish':
        x = x*keras.layers.ReLU(max_value=6.0)(x+3.0)/6.0
    elif non_linear == 'hsigmoid':
        x = keras.layers.ReLU(max_value=6.0)(x+3.0)/6.0
    elif non_linear == 'sigmoid':
        x = keras.activations.sigmoid(x)
    return x


def bottleneck(inputs, in_channels, expand_channels, out_channels, kernel_size, stride, se_channels=0, non_linear='hswish'):

    conv_ep = conv2d(inputs, filters=expand_channels, kernel_size=1, stride=1, padding='valid', non_linear=non_linear)  # expand

    conv_dw = conv2d(conv_ep, filters=expand_channels, kernel_size=kernel_size, stride=stride,
                     deepwise=True, non_linear=non_linear)  # depthwise

    outputs = conv2d(conv_dw, out_channels, 1, 1, padding='valid', non_linear='none')

    if se_channels:  # SE
        avgpool = keras.layers.GlobalAveragePooling2D(data_format='channels_last')(outputs)
        avgpool = keras.layers.Reshape((1, 1, out_channels))(avgpool)
        conv_s = conv2d(avgpool, se_channels, kernel_size=1, stride=1, padding='valid', non_linear='relu')
        conv_e = conv2d(conv_s, out_channels, kernel_size=1, stride=1, padding='valid', non_linear='hsigmoid')

        outputs = keras.layers.Multiply()([outputs, conv_e])

    if stride == 1:
        if in_channels != out_channels:
            inputs = conv2d(inputs, filters=out_channels, kernel_size=1, stride=1, padding='valid', non_linear='none')
        outputs = outputs + inputs  # Residual

    return outputs


def MobilenetV3():
    inputs = keras.layers.Input(shape=(224, 224, 3))

    x = conv2d(inputs, filters=16, kernel_size=3, stride=2, non_linear='hswish')

    x = bottleneck(x, in_channels=16, expand_channels=16, out_channels=16, kernel_size=3, stride=2, se_channels=4, non_linear='relu')
    x = bottleneck(x, in_channels=16, expand_channels=72, out_channels=24, kernel_size=3, stride=2, se_channels=0, non_linear='relu')
    x = bottleneck(x, in_channels=24, expand_channels=88, out_channels=24, kernel_size=3, stride=1, se_channels=0, non_linear='relu')

    x = bottleneck(x, in_channels=24, expand_channels=96, out_channels=40, kernel_size=5, stride=2, se_channels=10, non_linear='hswish')
    x = bottleneck(x, in_channels=40, expand_channels=240, out_channels=40, kernel_size=5, stride=1, se_channels=10, non_linear='hswish')
    x = bottleneck(x, in_channels=40, expand_channels=240, out_channels=40, kernel_size=5, stride=1, se_channels=10, non_linear='hswish')
    x = bottleneck(x, in_channels=40, expand_channels=120, out_channels=48, kernel_size=5, stride=1, se_channels=12, non_linear='hswish')
    x = bottleneck(x, in_channels=48, expand_channels=144, out_channels=48, kernel_size=5, stride=1, se_channels=12, non_linear='hswish')
    x = bottleneck(x, in_channels=48, expand_channels=288, out_channels=96, kernel_size=5, stride=2, se_channels=24, non_linear='hswish')
    x = bottleneck(x, in_channels=96, expand_channels=576, out_channels=96, kernel_size=5, stride=1, se_channels=24, non_linear='hswish')
    x = bottleneck(x, in_channels=96, expand_channels=576, out_channels=96, kernel_size=5, stride=1, se_channels=24,non_linear='hswish')

    x = conv2d(x, filters=576, kernel_size=1, stride=1, non_linear='hswish')

    x = keras.layers.GlobalAveragePooling2D(data_format='channels_last')(x)

    x = keras.layers.Reshape((576,))(x)
    x = keras.layers.Dense(1280, activation=None)(x)
    x = keras.layers.BatchNormalization()(x)
    x = x * keras.layers.ReLU(max_value=6.0)(x + 3.0) / 6.0  # hswish
    outputs = keras.layers.Dense(1000, activation='softmax')(x)

    return keras.models.Model(inputs=[inputs], outputs=[outputs])


"""
model = MobilenetV3()
model.summary()
weight_path = "D:\data/tiny_nets\weights\mbv3\mbv3"
model.load_weights(weight_path)
model.training = False
img = cv2.imread("D:\data\image\ILSVRC2012_val_00000001.JPEG")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (224, 224))

img = tf.constant(img, dtype=tf.float32)
img /= 127.5
img -= 1.0
img = tf.expand_dims(img, 0)

pred = model(img)
print(tf.shape(pred))
print(tf.reduce_max(pred))
print(tf.argmax(pred, axis=1))
"""