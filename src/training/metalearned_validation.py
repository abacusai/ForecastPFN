import sys
ACADEMIC_HOME = '/home/ubuntu/notebooks/ForecastPFN/academic_comparison/'
METALEARNED_HOME = ACADEMIC_HOME + 'metalearned/'
sys.path.append(ACADEMIC_HOME)
sys.path.append(METALEARNED_HOME)

import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from data_provider.UnivariateTimeseriesSampler_WithStamps import UnivariateTimeseriesSampler_WithStamps
from resources.tourism.dataset import TourismDataset, TourismMeta
from resources.m3.dataset import M3Dataset, M3Meta

def _ForecastPFN_time_features(ts: np.ndarray):
    if type(ts[0]) == datetime.datetime:
        year = [x.year for x in ts]
        month = [x.month for x in ts]
        day = [x.day for x in ts]
        day_of_week = [x.weekday()+1 for x in ts]
        day_of_year = [x.timetuple().tm_yday for x in ts]
        return np.stack([year, month, day, day_of_week, day_of_year], axis=-1)
    ts = pd.to_datetime(ts)
    return np.stack([ts.year, ts.month, ts.day, ts.day_of_week + 1, ts.day_of_year], axis=-1)


def prepare_metalearned_test(metaleanredDataset, metalearnedMeta, sp, p_input_size, p_horizon) -> tf.data.Dataset:

    target_input, target_output = metaleanredDataset(
            METALEARNED_HOME+metalearnedMeta.dataset_path).standard_split()
    in_bundle, out_bundle, sp = target_input, target_output, sp
    in_bundle = in_bundle.filter(
        lambda ts: ts.meta['seasonal_pattern'] == sp)
    out_bundle = out_bundle.filter(
        lambda ts: ts.meta['seasonal_pattern'] == sp)
    

    input_set = in_bundle.values()
    input_timestamps = in_bundle.time_stamps()
    input_set = UnivariateTimeseriesSampler_WithStamps(timeseries=input_set,
                                                    time_stamps=input_timestamps,
                                                    insample_size=p_input_size,
                                                    outsample_size=0,
                                                    window_sampling_limit=1,
                                                    batch_size=1,
                                                    time_features=_ForecastPFN_time_features,
                                                    )
    p_x, p_x_mask, p_x_timestamps = input_set.sequential_latest_insamples()

    output_set = out_bundle.values()
    output_timestamps = out_bundle.time_stamps()
    output_set = UnivariateTimeseriesSampler_WithStamps(timeseries=output_set,
                                                        time_stamps=output_timestamps,
                                                        insample_size=p_horizon,
                                                        outsample_size=0,
                                                        window_sampling_limit=1,
                                                        batch_size=1,
                                                        time_features=_ForecastPFN_time_features,
                                                        )
    p_y, p_y_mask, p_y_timestamps = output_set.sequential_latest_insamples()

    x, x_mark, y, y_mark = p_x, p_x_timestamps, p_y, p_y_timestamps

    ts = []
    history = []
    target_ts = []
    task = []
    y_out = []
    for x, y, x_mark, y_mark in zip(p_x, p_y, p_x_timestamps, p_y_timestamps):
        for yi, yi_mark in zip(y,y_mark):
            if sum(yi_mark):
                ts.append(np.append(np.zeros((100 - x_mark.shape[0],5)), x_mark, axis=0))
                history.append(np.append(np.zeros(100 - x_mark.shape[0]), x))
                target_ts.append(np.array([yi_mark]))
                task.append(1)
                y_out.append([yi])

    ts = tf.convert_to_tensor(np.array(ts), dtype=np.int64, name='ts')
    history = tf.convert_to_tensor(np.array(history), dtype=np.float32, name='history')
    target_ts = tf.convert_to_tensor(np.array(target_ts), dtype=np.int64, name='target_ts')
    task = tf.convert_to_tensor(np.array(task), dtype=np.int64, name='task')
    y = tf.convert_to_tensor(np.array(y_out), dtype=np.float32)

    ds = {
                'ts': ts,
                'history': history,
                'target_ts': target_ts,
                'task': task
            }, y

    return tf.data.Dataset.from_tensor_slices(ds)


# Tourism
tourism_yearly_test_df = prepare_metalearned_test(
    TourismDataset, TourismMeta, 'Yearly', 8, 4) 
tourism_quarterly_test_df = prepare_metalearned_test(
    TourismDataset, TourismMeta, 'Quarterly', 16, 8) 
tourism_monthly_test_df = prepare_metalearned_test(
    TourismDataset, TourismMeta, 'Monthly', 48, 24) 

# M3
m3_yearly_test_df = prepare_metalearned_test(
    M3Dataset, M3Meta, 'M3Year', 12, 6) 
m3_quarterly_test_df = prepare_metalearned_test(
    M3Dataset, M3Meta, 'M3Quart', 16, 8) 
m3_monthly_test_df = prepare_metalearned_test(
    M3Dataset, M3Meta, 'M3Month', 36, 18) 
m3_others_test_df = prepare_metalearned_test(
    M3Dataset, M3Meta, 'M3Other', 16, 8) 
