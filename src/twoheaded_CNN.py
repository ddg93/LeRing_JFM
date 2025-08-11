#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 11 14:04:16 2025
@author: davide

Contains:
	- make_orientation_loss method to define a l2 loss function between true and predicted particle orientation vectors with customizable penalty on negative n3 predictions
	- build_two_head_LeNet to create a two-headed CNN model with concatenation and dense units at the end, inspired from the LeNet5 architecture,
	  for processing coupled side-top frames of an oriented object and regress its particle orientation vector.
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

def make_orientation_loss(penalty_weight=1.0):
    """
    Creates a custom orientation loss function with adjustable penalty_weight
    penalty on n3<0 predictions
    """
    def orientation_loss(y_true, y_pred):
        #mean squared norm-2 difference between vectors
        norm_loss = tf.norm(y_true - y_pred, axis=-1)
        loss = tf.reduce_mean(tf.square(norm_loss))
        #penalize negative (n3=z) forecasts
        penalty = tf.reduce_mean(tf.nn.relu(-y_pred[:, 2]))
        return loss + penalty_weight * penalty
    return orientation_loss

def build_two_head_LeNet(image_shape=(100, 100, 1), conv_filters=(8, 16, 32), conv_kernel_size=5, pool_size=2, 
                         activation='relu', padding='valid', dense_units=(256, 256, 256), 
                         l2_weight=0.001, dropout_rate=None):
    """
    Builds a two-headed LeNet-inspired CNN for particle orientation regression over coupled 
    (side/top) frames of an oriented object. Each head processes one view before their
    features are concatenated and the particle orientation vector n=(n_1,n_2,n_3) is regressed.
    The two heads are specified as equal.
    Parameters:
        - conv_filters is a list setting the number of convolutional layers and filters for each layer,
        - conv_kernel_size is the size of the convolutional kernels,
        - pool_size is the pooling filter size,
        - activation is the activation function,
        - padding sets the padding,
        - dense_units specify number of dense layers and number of neurons for each layer
        - l2_weight introduces l2 weight regularization on all the layers of the model
        - dropout_rate sets the drop-out rate for the dense layers, if not None
    """
    #side view head
    side_input = layers.Input(shape=image_shape, name="side_input")
    x = side_input
    for filters in conv_filters:
        x = layers.Conv2D(filters, kernel_size=conv_kernel_size, activation=activation,
                          strides=(1, 1), padding=padding,
                          kernel_regularizer=regularizers.l2(l2_weight))(x)
        x = layers.MaxPooling2D(pool_size=pool_size, strides=2)(x)

    #top view head
    top_input = layers.Input(shape=image_shape, name="top_input")
    y = top_input
    for filters in conv_filters:
        y = layers.Conv2D(filters, kernel_size=conv_kernel_size, activation=activation,
                          strides=(1, 1), padding=padding,
                          kernel_regularizer=regularizers.l2(l2_weight))(y)
        y = layers.MaxPooling2D(pool_size=pool_size, strides=2)(y)

    #concatenate the two branches
    z = layers.concatenate([x, y])
    z = layers.Flatten()(z)

    #dense layers
    for units in dense_units:
        z = layers.Dense(units, activation=activation,
                         kernel_regularizer=regularizers.l2(l2_weight))(z)
        if dropout_rate:
            z = layers.Dropout(rate=dropout_rate)(z)

    #output layer: 3D particle orientation vector with L2-norm=1 by design
    final_dense = layers.Dense(3, activation='linear')(z)
    output = layers.Lambda(lambda n: tf.math.l2_normalize(n, axis=-1, epsilon=1e-12))(final_dense)

    return models.Model(inputs=[side_input, top_input], outputs=output, name="CNN_Jeffery")
