"""
Module to prepare customer dataset for evaluation
"""
import numpy as np
import pandas as pd
import tensorflow as tf

HISTORY = 100


def compute_time_features(ts: np.ndarray):
    ts = pd.to_datetime(ts)
    return np.stack(
        [ts.year, ts.month, ts.day, ts.day_of_week + 1, ts.day_of_year], axis=-1
    )


def build_input(ts, target, task=1):
    horizon = len(ts) - len(target)
    all_dates = tf.numpy_function(compute_time_features, inp=[ts], Tout=tf.int64)
    date_tensor = all_dates[:-horizon]
    target_dates = all_dates[-horizon:]

    # this is the target value of the data before the horizon
    target = tf.convert_to_tensor(target, dtype=tf.float32)

    underflow = HISTORY - date_tensor.shape[0]
    if underflow > 0:
        target = tf.pad(target, [[underflow, 0]])
        date_tensor = tf.pad(date_tensor, [[underflow, 0], [0, 0]])

    # if date tensor was greater than the history desired, then trim it
    # to get the last HISTORY number of values
    date_tensor = date_tensor[-HISTORY:]
    return {
        "ts": tf.repeat(tf.expand_dims(date_tensor, axis=0), [horizon], axis=0),
        # repeat the before horizon values horizon number of times,
        # so that for each of the predictions for each target_ts, you
        # have an available set of features
        "history": tf.repeat(tf.expand_dims(target, axis=0), [horizon], axis=0),
        "target_ts": tf.expand_dims(target_dates, axis=1),
        "task": tf.fill(
            [
                horizon,
            ],
            task,
        ),
    }
