"""
Module to transform different real world datasets
into format used for our synthetic dataset
"""
import csv
from datetime import datetime
from functools import reduce

import numpy as np
import pandas as pd
import tensorflow as tf
from dateutil.relativedelta import relativedelta
from tqdm import trange

HISTORY = 100
HORIZON = 10
NUM_TASKS = 3


def compute_time_features(ts: np.ndarray):
    ts = pd.to_datetime(ts)
    return np.stack(
        [ts.year, ts.month, ts.day, ts.day_of_week + 1, ts.day_of_year], axis=-1
    )


def build_input(ts, target_full, task=1):
    # horizon should be fixed as defined in model
    all_dates = tf.numpy_function(compute_time_features, inp=[ts], Tout=tf.int64)
    date_tensor = all_dates[:-HORIZON]
    target_dates = all_dates[-HORIZON:]

    target = target_full[:-HORIZON]
    target_to_predict = target_full[-HORIZON:]

    if task == 2:
        target_to_predict = np.cumsum(target_to_predict) / (1 + np.arange(HORIZON))
    elif task == 3:
        target_to_predict = [
            np.std(target_to_predict[: i + 1]) for i in range(len(target_to_predict))
        ]

    # this is the target value of the data before the horizon
    target = tf.convert_to_tensor(target, dtype=tf.float32)

    underflow = HISTORY - date_tensor.shape[0]
    if underflow > 0:
        target = tf.pad(target, [[underflow, 0]])
        date_tensor = tf.pad(date_tensor, [[underflow, 0], [0, 0]])

    # if date tensor was greater than the history desired, then trim it
    # to get the last HISTORY number of values
    date_tensor = date_tensor[-HISTORY:]
    target = target[-HISTORY:]

    return {
        "ts": tf.repeat(tf.expand_dims(date_tensor, axis=0), [HORIZON], axis=0),
        # repeat the before horizon values horizon number of times,
        # so that for each of the predictions for each target_ts, you
        # have an available set of features
        "history": tf.repeat(tf.expand_dims(target, axis=0), [HORIZON], axis=0),
        "target_ts": tf.expand_dims(target_dates, axis=1),
        "task": tf.fill(
            [
                HORIZON,
            ],
            task,
        ),
    }, tf.expand_dims(tf.convert_to_tensor(target_to_predict, dtype=tf.float32), axis=1)


def read_timeseries_file(filename):
    """
    Function to read the standard datasets for time series.
    The datasets are in CSV format, hence the function is implemented
    accordingly
    """
    lines = []
    with open(filename) as fh:
        reader = csv.reader(fh)
        for line in reader:
            lines.append([float(x) for x in line])

    return lines


def get_dates(num_days, freq):
    dates = []

    # TODO: find better method for assigning these dates
    if freq == "daily":
        current_date = datetime(2020, 10, 10)
    elif freq == "monthly":
        current_date = datetime(2010, 1, 31)

    for _ in range(num_days):
        dates.append(pd.to_datetime(current_date))
        if freq == "daily":
            current_date += relativedelta(days=1)
        elif freq == "weekly":
            current_date += relativedelta(weeks=1)
        elif freq == "monthly":
            current_date += relativedelta(months=1)

    return dates


def split_dataset(dataset):
    """
    If the size of dataset is n * (HISTORY + HORIZON), we split it
    into 3*n different datasets, with HISTORY/3 overlap between them
    """
    mini_datasets = []
    i = 0

    # if the size of dataset is less than HISTORY + HORIZON
    # then take whatever points are available
    # otherwise, slide a window starting from the first point
    # with a stride of HISTORY // 3 until the elements in
    # window are less than HISTORY + HORIZON
    while i == 0 or i + HISTORY + HORIZON < len(dataset):
        mini_datasets.append(dataset[i : i + HISTORY + HORIZON])
        i += HISTORY // 3

    return mini_datasets


def build_dataset(dataset, freq):
    """
    Function to build dataframe for a training dataset of given frequency
    """
    ts_list, history_list, target_ts_list, task_list = [], [], [], []
    outputs = []

    # TODO: change it from 2
    # keeping it 2 for testing, as dataset creation takes time
    for i in trange(100):
        # for i in trange(len(dataset)):
        for X in split_dataset(dataset[i]):
            dates = get_dates(len(X), freq)

            for task in range(1, NUM_TASKS + 1):
                built_input, output = build_input(dates, X, task=task)

                ts_list += [ts for ts in built_input["ts"]]
                history_list += [history for history in built_input["history"]]
                target_ts_list += [target_ts for target_ts in built_input["target_ts"]]
                task_list += [task for task in built_input["task"]]

                outputs += [y for y in output]

    dataset_frame = tf.data.Dataset.from_tensor_slices(
        (
            {
                "ts": ts_list,
                "history": history_list,
                "target_ts": target_ts_list,
                "task": task_list,
            },
            outputs,
        )
    )

    return dataset_frame


def construct_dataframe(train_dataset_and_freq):
    """
    Function to construct the dataframe in accordance with the training format
    """
    dfs = []
    for dataset, freq in train_dataset_and_freq:
        dfs.append(build_dataset(dataset, freq))

    return reduce(lambda df1, df2: df1.concatenate(df2), dfs)


def get_validation_dataset():
    """
    Function to read data from various sources and feed them as input to
    build a dataframe for getting the validation dataset
    """
    wikiweb_train = read_timeseries_file(
        "/home/ubuntu/notebooks/forecasting/pretraining/wikiweb_train.csv"
    )
    tourism_train = read_timeseries_file(
        "/home/ubuntu/notebooks/forecasting/pretraining/tourism_train.csv"
    )
    read_timeseries_file(
        "/home/ubuntu/notebooks/forecasting/pretraining/exchange_rate_train.csv"
    )
    read_timeseries_file("/home/ubuntu/notebooks/forecasting/pretraining/m3_train.csv")

    # add different datasets and their frequency here
    # TODO: addition of monthly dataset shoots up
    # validation loss to ~40k. Need to see how to fix that
    train_dataset_and_freq = [
        (wikiweb_train, "daily"),
        (tourism_train, "monthly"),
        # (exchange_rate_train, "daily"),
        # (m3_train, "monthly")
    ]

    constructed_dataframe = construct_dataframe(train_dataset_and_freq)
    # print(len(list(constructed_dataframe)))
    return constructed_dataframe


def main():
    get_validation_dataset()


if __name__ == "__main__":
    main()
