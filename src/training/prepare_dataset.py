"""
Module to process data to convert it into form usable by model tasks
such as point prediction, mean prediction and std prediction
"""

from typing import Dict
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_io
from config_variables import Config
from constants import PADDING, HISTORY_LEN, TARGET_LEN, TRIM_LEN, TARGET_INDEX, \
    SINGLE_POINT, MEAN_TO_DATE, STDEV_TO_DATE


def compute_time_features(ts: np.ndarray):
    """
    Method to compute time features to be used by model
    :param ts: array consisting of int64Index representing timestamps
    :return: numpy array of shape (n, 5) containing the time features
    """
    ts = pd.to_datetime(ts)
    if Config.is_sub_day:
        return np.stack([ts.minute, ts.hour, ts.day, ts.day_of_week + 1, ts.day_of_year], axis=-1)
    return np.stack([ts.year, ts.month, ts.day, ts.day_of_week + 1, ts.day_of_year], axis=-1)



@tf.function
def build_frames(r: Dict[str, tf.Tensor]):
    raw_date_info = tf.numpy_function(compute_time_features, inp=[r['ts']], Tout=tf.int64)
    date_info = tf.signal.frame(
                    tf.pad(raw_date_info, [[PADDING, 0], [0, 0]]),
                    HISTORY_LEN,
                    1,
                    axis=0
                )


    history = tf.signal.frame(tf.pad(r['y'], [[PADDING, 0]]), HISTORY_LEN, 1, axis=-1)
    noise = tf.signal.frame(tf.pad(r['noise'], [[PADDING, 0]]), HISTORY_LEN, 1, axis=-1)

    target_dates = tf.signal.frame(raw_date_info, TARGET_LEN, 1, axis=0)
    target_values = tf.signal.frame(r['y'], TARGET_LEN, 1, axis=-1)
    target_noise = tf.signal.frame(r['noise'], TARGET_LEN, 1, axis=-1)

    start_index = target_values.shape[0] - TRIM_LEN
    batch_size = start_index - TARGET_LEN

    return (
            date_info[-start_index:-TARGET_LEN],
            history[-start_index:-TARGET_LEN],
            noise[-start_index:-TARGET_LEN],
            target_dates[TARGET_INDEX:],
            target_values[TARGET_INDEX:],
            target_noise[TARGET_INDEX:]
        )


@tf.function
def gen_random_single_point(
        date_info: tf.Tensor,
        history: tf.Tensor,
        noise: tf.Tensor,
        target_dates: tf.Tensor,
        target_values: tf.Tensor,
        target_noise: tf.Tensor
    ):


    # To limit to a single date
    batch_size = tf.shape(target_dates)[0]
    targets = tf.random.uniform(shape=[batch_size, 1], maxval=TARGET_LEN, dtype=tf.int32)
    target_date = tf.gather(target_dates, targets, axis=1, batch_dims=1)
    target_value = tf.gather(target_values, targets, axis=1, batch_dims=1)
    return dict(
        ts=date_info,
        history=history*noise,
        noise=noise,
        target_ts=target_date,
        task=tf.fill([batch_size,], SINGLE_POINT),
        target_noise=target_noise
    ), target_value


@tf.function
def gen_mean_to_random_date(
        date_info: tf.Tensor,
        history: tf.Tensor,
        noise: tf.Tensor,
        target_dates: tf.Tensor,
        target_values: tf.Tensor,
        target_noise: tf.Tensor
    ):
    # To limit to a single date
    batch_size = tf.shape(target_dates)[0]
    targets = tf.random.uniform(shape=[batch_size, 1], maxval=TARGET_LEN, dtype=tf.int32)
    target_date = tf.gather(target_dates, targets, axis=1, batch_dims=1)
    target_value = tf.math.reduce_mean(
                        tf.RaggedTensor.from_tensor(target_values, lengths=(targets[:, 0] + 1)),
                        keepdims=True,
                        axis=-1
                   )
    return dict(
        ts=date_info,
        history=history*noise*.75,
        noise=noise,
        target_ts=target_date,
        task=tf.fill([batch_size,], MEAN_TO_DATE),
        target_noise=target_noise
    ), target_value


@tf.function
def gen_std_to_random_date(
            date_info: tf.Tensor,
            history: tf.Tensor,
            noise: tf.Tensor,
            target_dates: tf.Tensor,
            target_values: tf.Tensor,
            target_noise: tf.Tensor
        ):
    # To limit to a single date
    batch_size = tf.shape(target_dates)[0]
    targets = tf.random.uniform(shape=[batch_size, 1], minval=(TARGET_LEN // 2), maxval=TARGET_LEN, dtype=tf.int32)
    target_date = tf.gather(target_dates, targets, axis=1, batch_dims=1)
    target_value = tf.math.reduce_std(
                        tf.RaggedTensor.from_tensor(target_values, lengths=(targets[:, 0] + 1)),
                        keepdims=True,
                        axis=-1
                   )
    target_noise_std = tf.math.reduce_std(
                        tf.RaggedTensor.from_tensor(target_noise, lengths=(targets[:, 0] + 1)),
                        keepdims=True,
                        axis=-1
                   )

    target_value = tf.math.sqrt(target_value**2 + target_noise_std**2)

    return dict(
        ts=date_info,
        history=history*noise,
        noise=noise,
        target_ts=target_date,
        task=tf.fill([batch_size,], STDEV_TO_DATE),
        target_noise=target_noise
    ), target_value

@tf.function
def gen_random_single_point_no_noise(
        date_info: tf.Tensor,
        history: tf.Tensor,
        noise: tf.Tensor,
        target_dates: tf.Tensor,
        target_values: tf.Tensor,
        target_noise: tf.Tensor
    ):


    # To limit to a single date
    batch_size = tf.shape(target_dates)[0]
    targets = tf.random.uniform(shape=[batch_size, 1], maxval=TARGET_LEN, dtype=tf.int32)
    target_date = tf.gather(target_dates, targets, axis=1, batch_dims=1)
    target_value = tf.gather(target_values, targets, axis=1, batch_dims=1)
    return dict(
        ts=date_info,
        history=history,
        target_ts=target_date,
        task=tf.fill([batch_size,], SINGLE_POINT),
    ), target_value


@tf.function
def gen_mean_to_random_date_no_noise(
        date_info: tf.Tensor,
        history: tf.Tensor,
        noise: tf.Tensor,
        target_dates: tf.Tensor,
        target_values: tf.Tensor,
        target_noise: tf.Tensor
    ):
    # To limit to a single date
    batch_size = tf.shape(target_dates)[0]
    targets = tf.random.uniform(shape=[batch_size, 1], maxval=TARGET_LEN, dtype=tf.int32)
    target_date = tf.gather(target_dates, targets, axis=1, batch_dims=1)
    target_value = tf.math.reduce_mean(
                        tf.RaggedTensor.from_tensor(target_values, lengths=(targets[:, 0] + 1)),
                        keepdims=True,
                        axis=-1
                   )
    return dict(
        ts=date_info,
        history=history,
        target_ts=target_date,
        task=tf.fill([batch_size,], MEAN_TO_DATE),
    ), target_value


@tf.function
def gen_std_to_random_date_no_noise(
            date_info: tf.Tensor,
            history: tf.Tensor,
            noise: tf.Tensor,
            target_dates: tf.Tensor,
            target_values: tf.Tensor,
            target_noise: tf.Tensor
        ):
    # To limit to a single date
    batch_size = tf.shape(target_dates)[0]
    targets = tf.random.uniform(shape=[batch_size, 1], minval=(TARGET_LEN // 2), maxval=TARGET_LEN, dtype=tf.int32)
    target_date = tf.gather(target_dates, targets, axis=1, batch_dims=1)
    target_value = tf.math.reduce_std(
                        tf.RaggedTensor.from_tensor(target_values, lengths=(targets[:, 0] + 1)),
                        keepdims=True,
                        axis=-1
                   )
    target_noise_std = tf.math.reduce_std(
                        tf.RaggedTensor.from_tensor(target_noise, lengths=(targets[:, 0] + 1)),
                        keepdims=True,
                        axis=-1
                   )

    target_value = tf.math.sqrt(target_value**2 + target_noise_std**2)

    return dict(
        ts=date_info,
        history=history,
        target_ts=target_date,
        task=tf.fill([batch_size,], STDEV_TO_DATE),
    ), target_value

@tf.function
def filter_unusable_points(X: Dict[str, tf.Tensor], y: tf.Tensor):
    """
    Filter points where the maximum in the history is less than 0.1
    """
    return tf.logical_and(tf.reduce_max(X['history']) > 0.1, tf.math.is_finite(y))[0]

def position_encoding(periods: int, freqs: int):
    return np.hstack([
        np.fromfunction(lambda i, j: np.sin(np.pi / periods * (2**j) * (i-1)), (periods + 1, freqs)),
        np.fromfunction(lambda i, j: np.cos(np.pi / periods * (2**j) * (i-1)), (periods + 1, freqs))
    ])