"""
Module to evaluate on the customer dataset
"""

import yaml
import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_io
from reainternal import environment
import reainternal.mllibs.pipelinelib as PL
from scipy.stats.mstats import winsorize
from tqdm import trange
from prepare_dataset import build_input


def evaluate(config):
    pretrained = tf.keras.models.load_model(config['model_path'])
    model_info = PL.get_model_info(config['model_number'])

    record_index = model_info.prepared_dataset_instance.get_record_index(3)

    # the string 'LastDay_Month'
    ts_col = model_info.serving_dataset_instance.config.timestamp_column
    # the string 'target'
    target_col = model_info.serving_dataset_instance.config.target_column

    BATCH_SIZE = 100
    item_id, pred_start, actual, pred = [], [], [], []
    for i in trange(0, len(record_index), BATCH_SIZE):
        test_points = list(model_info.prepared_dataset_instance.get_prediction_records(record_index[i:(i + BATCH_SIZE)]))
        for current_point in test_points:
            # contains the history of available values and the targets
            prediction_record, _ = model_info.serving_dataset_instance.dataset_class.prepare_data_for_prediction(
                model_info.serving_dataset_instance,
                current_point.model_input)

            # timestamps of history
            history_ts = prediction_record[ts_col]

            # values of history
            history = prediction_record[target_col]

            # mean of history's last 6 values
            history_mean = np.nanmean(history[-6:])

            # std of history's last 6 values
            history_std = np.nanstd(history[-6:])

            # local scale, don't know why defined so
            local_scale = (history_mean + history_std + 1e-4)

            # change history based on local scale, to normalize it between 0 and 1
            history = np.clip(history / local_scale, a_min=0, a_max=1)

            # get predicted mean based on this local scale
            pred_mean = pretrained(build_input(history_ts, history, task=2))

            # get scaled mean based on the given history
            scaled_mean = (pred_mean['result'].numpy().reshape(-1) * pred_mean['scale'].numpy().reshape(-1)) * local_scale

            item_id.append(current_point.test_info[0])
            pred_start.append(current_point.test_info[1])
            actual.append(np.mean(current_point.actual[target_col]))
            pred.append(scaled_mean[-1])

    eval_clipped_df = pd.DataFrame(dict(
        item_id=item_id,
        pred_start=pred_start,
        actual=actual,
        pred=pred
    ))

    eval_clipped_df = eval_clipped_df.assign(
        cmape=lambda df: np.abs(df.actual - df.pred) / df.actual
    ).assign(
        winsorized_cmape=lambda df: winsorize(df.cmape, (0.01, 0.01)),
        squashed_cmape=lambda df: np.where(df.cmape > 1, 1 + np.log(df.cmape), df.cmape)
    )

    return eval_clipped_df[(eval_clipped_df.actual > 0) & (eval_clipped_df.pred_start == '2021-06-30T00:00:00')].describe()



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, help="Path to config file")
    args = parser.parse_args()

    with open(args.config) as config_file:
        config = yaml.load(config_file, yaml.loader.SafeLoader)

    environment.change_config_env('jenkinsdev')

    results_df = evaluate(config)
    print(results_df)

if __name__ == '__main__':
    main()