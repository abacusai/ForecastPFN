"""
Module containing different scaler functions
"""
import tensorflow as tf
import tensorflow_io
from tensorflow.keras import layers
import numpy as np

def robust_scaler(inputs, epsilon):
    # inputs.shape = (batch_size, history_len, 1)
    # masking is implemented here to not include the points having
    # missing data into calculation of mean and std, as that causes
    # unwanted skewness in data. To find missing points we compare
    # the inputs with 0

    scaling_context = inputs
    scaling_mask = inputs != 0

    non_zero_inputs = tf.ragged.boolean_mask(scaling_context, scaling_mask)

    # mean and std = (batch_size, 1, 1)
    mean = tf.math.reduce_mean(non_zero_inputs, axis=1, keepdims=True).to_tensor()
    std = tf.math.reduce_std(non_zero_inputs, axis=1, keepdims=True).to_tensor()

    # clipped.shape = (batch_size, history_len, 1)
    clipped = tf.clip_by_value(inputs, 0, mean + 2 * std)

    clipped_and_masked = tf.ragged.boolean_mask(clipped, (clipped != 0))

    # calculate mean and std of clipped data
    clipped_mean = tf.math.reduce_mean(
        clipped_and_masked,
        axis=1,
        keepdims=True
    ).to_tensor()
    clipped_std = tf.math.reduce_std(
        clipped_and_masked,
        axis=1,
        keepdims=True
    ).to_tensor()

    # scale is of shape (batch_size,1,1)
    scale = clipped_mean + clipped_std + epsilon
    inputs_scaled = inputs / scale

    # clip the values to prevent exploding gradients
    # 3 is chosen as a rough estimate for the upper bound of clipping
    inputs_clipped = tf.clip_by_value(inputs_scaled, 0, 3)

    return scale, inputs_clipped


def max_scaling(inputs, epsilon):

    scaler = layers.GlobalMaxPooling1D(name='MaxScaling', keepdims=1)

    scale = scaler(inputs) + epsilon
    output = inputs / scale
    return scale, output

