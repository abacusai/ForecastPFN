from tqdm import tqdm
import pathlib
import sys
sys.path.append('..')

import logging
import os

import numpy as np
import pandas as pd
import torch as t
import tensorflow as tf
import tensorflow_io
from fire import Fire
from scipy.interpolate import interp1d
from torch import optim

from common.experiment import create_experiment
from common.experiment import load_experiment_parameters
from data_provider.UnivariateTimeseriesSampler_WithStamps import UnivariateTimeseriesSampler_WithStamps
from exp.exp_ForecastPFN import Exp_ForecastPFN
from common.settings import experiment_path
from common.timeseries import TimeseriesBundle
from common.torch_utils import SnapshotManager, to_device, to_tensor, mase_loss, mape_loss, smape_2_loss
from common.utils import get_module_path
from common.metrics import smape
from experiments.tl.parameters import parameters
from models.nbeats_torch import nbeats_generic, nbeats_interpretable
from resources.m3.dataset import M3Dataset, M3Meta
from resources.m4.dataset import M4Dataset, M4Meta
from resources.tourism.dataset import TourismDataset, TourismMeta

module_path = get_module_path()


def init(name: str):
    create_experiment(experiment_path=experiment_path(module_path, name),
                      parameters=parameters[name],
                      command=lambda path, params: f'python {module_path}/main.py run --path={path}')


def run(path: str):
    experiment_parameters = load_experiment_parameters(path)

    model_horizons = {
        'Y4': 4,
        'Y6': 6,
        'Q8': 8,
        'M18': 18,
        'M24': 24,
        'W13': 13,
        'D14': 14,
        'H24': 24,
        'H48': 48,
    }

    tl_models = {}
    for model_name, horizon in model_horizons.items():
        input_size = experiment_parameters['lookback_period'] * horizon
        model = Exp_ForecastPFN(None)

        tl_models[model_name] = {'p_model': model, 'p_input_size': input_size, 'p_horizon': horizon}

    #
    # Predictions
    #

    def forecast(in_bundle: TimeseriesBundle, out_bundle: TimeseriesBundle,
                 sp: str,
                 p_model, p_input_size, p_horizon):
        forecasts = []

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
                                                           time_features=p_model._ForecastPFN_time_features,
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
                                                            time_features=p_model._ForecastPFN_time_features,
                                                            )
        p_y, p_y_mask, p_y_timestamps = output_set.sequential_latest_insamples()

        x, x_mark, y, y_mark = p_x, p_x_timestamps, p_y, p_y_timestamps

        batch_x, batch_y = to_tensor(x)[:, :, None], to_tensor(y)[:, :, None]
        batch_x_mark, batch_y_mark = to_tensor(
            x_mark.astype(int)), to_tensor(y_mark.astype(int))
    
        
        model = tf.keras.models.load_model(
            str(pathlib.Path(path).parent) + '/ckpts/', custom_objects={'smape': smape})
        for idx, (x, y, x_mark, y_mark) in tqdm(enumerate(zip(batch_x, batch_y, batch_x_mark, batch_y_mark))):
            pred = p_model._process_tuple(x, x_mark, y_mark, model, p_horizon)
            forecasts.extend(pred)

        forecasts_df = pd.DataFrame(forecasts, columns=[f'V{idx + 1}' for idx in range(p_horizon)])
        forecasts_df.index = in_bundle.ids()
        forecasts_df.index.name = 'id'
        return forecasts_df


    # M4
    # target_input, target_output = M4Dataset(
    #     M4Meta.dataset_path).standard_split()
    # yearly = forecast(target_input, target_output, 'Yearly', **tl_models['Y6'])
    # quarterly = forecast(target_input, target_output, 'Quarterly', **tl_models['Q8'])
    # monthly = forecast(target_input, target_output, 'Monthly', **tl_models['M18'])
    # weekly = forecast(target_input, target_output, 'Weekly', **tl_models['W13'])
    # daily = forecast(target_input, target_output, 'Daily', **tl_models['D14'])
    # hourly = forecast(target_input, target_output, 'Hourly', **tl_models['H48'])
    # pd.concat([yearly, quarterly, monthly, weekly, daily, hourly], sort=False).to_csv(
    #     os.path.join(os.path.join(path, 'M4.csv')))

    # M3
    target_input, target_output = M3Dataset(
        M3Meta.dataset_path).standard_split()
    yearly = forecast(target_input, target_output, 'M3Year', **tl_models['Y6'])
    quarterly = forecast(target_input, target_output, 'M3Quart', **tl_models['Q8'])
    monthly = forecast(target_input, target_output, 'M3Month', **tl_models['M18'])
    others = forecast(target_input, target_output, 'M3Other', **tl_models['Q8'])
    pd.concat([yearly, quarterly, monthly, others], sort=False).to_csv(os.path.join(os.path.join(path, 'M3.csv')))

    # Tourism
    target_input, target_output = TourismDataset(TourismMeta.dataset_path).standard_split()
    yearly = forecast(target_input, target_output, 'Yearly', **tl_models['Y4'])
    quarterly = forecast(target_input, target_output, 'Quarterly', **tl_models['Q8'])
    monthly = forecast(target_input, target_output, 'Monthly', **tl_models['M24'])
    pd.concat([yearly, quarterly, monthly], sort=False).to_csv(os.path.join(os.path.join(path, 'tourism.csv')))

def evaluate(name: str, summary_filter: str, validation_mode: bool = False):
    pass


def summary(name: str, summary_filter: str = '*', validation_mode: bool = False):
    pass


if __name__ == '__main__':
    logging.root.setLevel(logging.INFO)
    Fire()
