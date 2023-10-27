"""
Module to evaluate the model on real world datasets
"""
import yaml
import argparse
import tensorflow as tf
import tensorflow_io
import pandas as pd
import numpy as np
from process_data import read_timeseries_file
from tqdm import trange
from scipy.stats.mstats import winsorize
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler


HISTORY = 100

def compute_time_features(ts: np.ndarray):
    ts = pd.to_datetime(ts)
    return np.stack([ts.year, ts.month, ts.day, ts.day_of_week + 1, ts.day_of_year], axis=-1)
    # return np.stack([ts.minute, ts.hour, ts.day, ts.day_of_week + 1, ts.day_of_year], axis=-1)

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
    target = target[-HISTORY:]
    return {
        'ts': tf.repeat(tf.expand_dims(date_tensor, axis=0), [horizon], axis=0),

        # repeat the before horizon values horizon number of times,
        # so that for each of the predictions for each target_ts, you
        # have an available set of features
        'history': tf.repeat(tf.expand_dims(target, axis=0), [horizon], axis=0),
        'target_ts': tf.expand_dims(target_dates, axis=1),
        'task': tf.fill([horizon,], task),
    }

def evaluate_model(config, train_data, test_data, freq, name):
    pretrained = tf.keras.models.load_model(config['model_path'])

    BATCH_SIZE = 100
    item_id, pred_start, actual, pred = [], [], [], []
    stds = []
    wapes = []
    for i in trange(0, len(train_data), BATCH_SIZE):
        test_points = train_data[i:(i+BATCH_SIZE)]
        for idx, current_point in enumerate(test_points):

            # timestamps of history
            history_ts = pd.date_range(start='2010-01-01', periods=len(train_data[i+idx] + test_data[i+idx]), freq=freq)

            # values of history
            history = train_data[i+idx]

            # mean of history's last 6 values
            history_mean = np.nanmean(history[-6:])

            # std of history's last 6 values
            history_std = np.nanstd(history[-6:])

            # local scale, don't know why defined so
            local_scale = (history_mean + history_std + 1e-4)

            # change history based on local scale, to normalize it between 0 and 1
            history = np.clip(history / local_scale, a_min=0, a_max=1)

            # skip this point if the last 100 points had the value 0
            if np.max(history[-HISTORY:]) == 0:
                continue

            # get predicted mean based on this local scale
            pred_vals = pretrained(build_input(history_ts, history, task=1))

            # get scaled mean based on the given history
            scaled_vals = (pred_vals['result'].numpy().reshape(-1) * pred_vals['scale'].numpy().reshape(-1)) * local_scale

            if np.mean(np.array(test_data[i+idx])):
                wape = np.mean(np.abs(scaled_vals - np.array(test_data[i+idx]))) / np.mean(np.array(test_data[i+idx]))
                wapes.append(wape)

            assert len(scaled_vals) == len(test_data[i+idx])

            scaler = MinMaxScaler()
            scaler.fit(np.array(train_data[i+idx]).reshape(-1, 1))

            predicted_scaled = scaler.transform(np.array(scaled_vals).reshape(-1, 1))
            actual_scaled = scaler.transform(np.array(test_data[i+idx]).reshape(-1, 1))
            stds.append(np.std(actual_scaled))

            for pred_val, actual_val in zip(predicted_scaled, actual_scaled):
                pred_val, actual_val = pred_val[0], actual_val[0]
                if np.isfinite(pred_val):
                    pred.append(pred_val)
                    actual.append(actual_val)



    eval_clipped_df = pd.DataFrame(dict(
        actual=actual,
        pred=pred
    ))

    eval_clipped_df = eval_clipped_df.assign(
        cmape=lambda df: np.abs(df.actual - df.pred) / df.actual
    ).assign(
        winsorized_cmape=lambda df: winsorize(df.cmape, (0.01, 0.01)),
        squashed_cmape=lambda df: np.where(df.cmape > 1, 1 + np.log(df.cmape), df.cmape)
    )

    print(eval_clipped_df[(eval_clipped_df.actual > 0)].describe())

    # print(wapes)
    # print(np.nanmean(wapes))
    print("MAE:", mean_absolute_error(actual, pred))
    print("MSE:", mean_squared_error(actual, pred))
    print(np.mean(stds))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, help="Path to config file")
    args = parser.parse_args()

    with open(args.config) as config_file:
        config = yaml.load(config_file, yaml.loader.SafeLoader)

    train_data = read_timeseries_file(config['train_file'])
    test_data = read_timeseries_file(config['test_file'])


    evaluate_model(config, train_data, test_data, config['freq'], config['name'])



if __name__ == '__main__':
    main()
