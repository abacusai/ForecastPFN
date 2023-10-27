import logging
import os

import numpy as np
import pandas as pd
import torch as t
from fire import Fire
from scipy.interpolate import interp1d
from torch import optim

from common.experiment import create_experiment
from common.experiment import load_experiment_parameters
from common.samplers import UnivariateTimeseriesSampler
from common.settings import experiment_path
from common.timeseries import TimeseriesBundle
from common.torch_utils import SnapshotManager, to_device, to_tensor, mase_loss, mape_loss, smape_2_loss
from common.utils import get_module_path
from experiments.tl.parameters import parameters
from models.nbeats_torch import nbeats_generic, nbeats_interpretable
from resources.electricity.dataset import ElectricityDataset, ElectricityMeta
from resources.fred.dataset import FredDataset, FredMeta
from resources.m3.dataset import M3Dataset, M3Meta
from resources.m4.dataset import M4Dataset, M4Meta
from resources.tourism.dataset import TourismDataset, TourismMeta
from resources.traffic.dataset import TrafficDataset, TrafficMeta

module_path = get_module_path()


def init(name: str):
    create_experiment(experiment_path=experiment_path(module_path, name),
                      parameters=parameters[name],
                      command=lambda path, params: f'python {module_path}/main.py run --path={path}')


def run(path: str):
    experiment_parameters = load_experiment_parameters(path)
    source_dataset_name = experiment_parameters['source_dataset'] if 'source_dataset' in experiment_parameters else 'M4'
    loss_name = experiment_parameters['loss_name']

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

    if source_dataset_name == 'M4':
        source_dataset, _ = M4Dataset(M4Meta.dataset_path).standard_split()
        model_sps = {
            'Y4': 'Yearly',
            'Y6': 'Yearly',
            'Q8': 'Quarterly',
            'M18': 'Monthly',
            'M24': 'Monthly',
            'W13': 'Weekly',
            'D14': 'Daily',
            'H24': 'Hourly',
            'H48': 'Hourly',
        }
    elif source_dataset_name == 'FRED':
        source_dataset, _ = FredDataset(FredMeta.dataset_path).standard_split()
        model_sps = {
            'Y4': 'Yearly',
            'Y6': 'Yearly',
            'Q8': 'Quarterly',
            'M18': 'Monthly',
            'M24': 'Monthly',
            'W13': 'Monthly',
            'D14': 'Monthly',
            'H24': 'Monthly',
            'H48': 'Monthly',
        }
    elif source_dataset_name == 'M3':
        source_dataset, _ = M3Dataset(M3Meta.dataset_path).standard_split()
        model_sps = {
            'Y4': 'M3Year',
            'Y6': 'M3Year',
            'Q8': 'M3Quart',
            'M18': 'M3Month',
            'M24': 'M3Month',
            'W13': 'M3Month',
            'D14': 'M3Other',
            'H24': 'M3Other',
            'H48': 'M3Other',
        }
    else:
        raise Exception(f'Unknown source dataset {source_dataset_name}')

    tl_models = {}
    for model_name, horizon in model_horizons.items():
        sp = model_sps[model_name]
        training_subset = source_dataset.filter(lambda ts: ts.meta['seasonal_pattern'] == sp)
        training_values = np.array(training_subset.values())
        if source_dataset_name == 'FRED':  # interpolate monthly data
            if model_name == 'H24':
                training_values = []
                for values in training_subset.values():
                    interpolation_fn = interp1d(x=np.array(range(len(values))), y=values, kind='linear')
                    training_values.append(interpolation_fn(np.arange(0, len(values) - 0.5, 0.5)))
                training_values = np.array(training_values)
            elif model_name == 'H48':
                training_values = []
                for values in training_subset.values():
                    interpolation_fn = interp1d(x=np.array(range(len(values))), y=values, kind='linear')
                    training_values.append(interpolation_fn(np.arange(0, len(values) - 0.75, 0.25)))
                training_values = np.array(training_values)

        input_size = experiment_parameters['lookback_period'] * horizon
        training_dataset = UnivariateTimeseriesSampler(timeseries=training_values,
                                                       insample_size=input_size,
                                                       outsample_size=horizon,
                                                       window_sampling_limit=int(
                                                           experiment_parameters['history_horizons'] * horizon),
                                                       batch_size=experiment_parameters['batch_size'])

        #
        # Training
        #
        snapshot_dir = os.path.join(path, 'snapshots', model_name)

        snapshot_manager = SnapshotManager(snapshot_dir=snapshot_dir,
                                           logging_frequency=experiment_parameters['logging_frequency'],
                                           snapshot_frequency=experiment_parameters['snapshot_frequency'])

        if experiment_parameters['model_type'] == 'generic':
            model = nbeats_generic(input_size=input_size,
                                   output_size=horizon,
                                   blocks=experiment_parameters['blocks'],
                                   stacks=experiment_parameters['stacks'],
                                   fc_layers=experiment_parameters['layers'],
                                   fc_layers_size=experiment_parameters['width'],
                                   scaling=experiment_parameters['scaling'],
                                   mode=experiment_parameters['mode'])
        else:
            model = nbeats_interpretable(input_size=input_size,
                                         output_size=horizon,
                                         trend_blocks=experiment_parameters['trend_blocks'],
                                         trend_fc_layers=experiment_parameters['layers'],
                                         trend_fc_layers_size=experiment_parameters['trend_fc_layers_size'],
                                         degree_of_polynomial=experiment_parameters['degree_of_polynomial'],
                                         seasonality_blocks=experiment_parameters['seasonality_blocks'],
                                         seasonality_fc_layers=experiment_parameters['layers'],
                                         seasonality_fc_layers_size=experiment_parameters['seasonality_fc_layers_size'],
                                         num_of_harmonics=experiment_parameters['num_of_harmonics'],
                                         scaling=experiment_parameters['scaling'],
                                         mode=experiment_parameters['mode'])

        model = to_device(model)

        optimizer = optim.Adam(model.parameters(),
                               lr=experiment_parameters['learning_rate'],
                               weight_decay=0.0)

        lr_decay_step = experiment_parameters['iterations'] // 3
        if lr_decay_step == 0:
            lr_decay_step = 1

        iteration = snapshot_manager.restore(model, optimizer)

        #
        # Training Loop
        #
        snapshot_manager.enable_time_tracking()
        training_set = iter(training_dataset)
        for i in range(iteration + 1, experiment_parameters['iterations'] + 1):
            model.train()
            x, x_mask, y, y_mask = map(to_tensor, next(training_set))
            optimizer.zero_grad()
            forecast = model(x, x_mask)
            if loss_name == 'MAPE':
                training_loss = mape_loss(forecast, y, y_mask)
            elif loss_name == 'MASE':
                training_loss = mase_loss(x, training_subset.timeseries[0].period, forecast, y, y_mask)
            elif loss_name == 'SMAPE':
                training_loss = smape_2_loss(forecast, y, y_mask)
            else:
                raise Exception(f'Unknown loss function: {loss_name}')

            if np.isnan(float(training_loss)):
                break

            training_loss.backward()
            t.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            for param_group in optimizer.param_groups:
                param_group['lr'] = experiment_parameters['learning_rate'] * 0.5 ** (i // lr_decay_step)

            snapshot_manager.register(iteration=i,
                                      training_loss=float(training_loss),
                                      validation_loss=np.nan, model=model,
                                      optimizer=optimizer)
        tl_models[model_name] = {'p_model': model, 'p_input_size': input_size, 'p_horizon': horizon}

    #
    # Predictions
    #

    def forecast(bundle: TimeseriesBundle, p_model, p_input_size, p_horizon):
        forecasts = []
        input_set = np.array(bundle.values())
        input_set = UnivariateTimeseriesSampler(timeseries=input_set,
                                                insample_size=p_input_size,
                                                outsample_size=0,
                                                window_sampling_limit=1,
                                                batch_size=1)
        p_x, p_x_mask = map(to_tensor, input_set.sequential_latest_insamples())
        p_model.eval()
        with t.no_grad():
            forecasts.extend(p_model(p_x, p_x_mask).cpu().detach().numpy())

        forecasts_df = pd.DataFrame(forecasts, columns=[f'V{idx + 1}' for idx in range(p_horizon)])
        forecasts_df.index = bundle.ids()
        forecasts_df.index.name = 'id'
        return forecasts_df

    def rolling_daily_forecast(base_insample: TimeseriesBundle, rolling_insample: TimeseriesBundle,
                               p_model, p_input_size, p_horizon):
        forecasts = []
        base_insample_values = np.array(base_insample.values())
        rolling_insample_values = np.array(rolling_insample.values())
        for window_id in range(7):
            insample = np.concatenate([base_insample_values, rolling_insample_values[:, :window_id * p_horizon]],
                                      axis=1)
            input_set = UnivariateTimeseriesSampler(timeseries=insample,
                                                    insample_size=p_input_size,
                                                    outsample_size=0,
                                                    window_sampling_limit=1,
                                                    batch_size=1)
            p_x, p_x_mask = map(to_tensor, input_set.sequential_latest_insamples())
            p_model.eval()
            with t.no_grad():
                window_forecast = p_model(p_x, p_x_mask).cpu().detach().numpy()
                forecasts = window_forecast if len(forecasts) == 0 else np.concatenate([forecasts, window_forecast],
                                                                                       axis=1)

        forecasts_df = pd.DataFrame(forecasts, columns=[f'V{idx + 1}' for idx in range(p_horizon * 7)])
        forecasts_df.index = base_insample.ids()
        forecasts_df.index.name = 'id'
        forecasts_df.columns = [f'V{i}' for i in range(1, len(forecasts_df.columns) + 1)]
        return forecasts_df

    # M4
    target_input, _ = M4Dataset(M4Meta.dataset_path).standard_split()
    yearly = forecast(target_input.filter(lambda ts: ts.meta['seasonal_pattern'] == 'Yearly'), **tl_models['Y6'])
    quarterly = forecast(target_input.filter(lambda ts: ts.meta['seasonal_pattern'] == 'Quarterly'), **tl_models['Q8'])
    monthly = forecast(target_input.filter(lambda ts: ts.meta['seasonal_pattern'] == 'Monthly'), **tl_models['M18'])
    weekly = forecast(target_input.filter(lambda ts: ts.meta['seasonal_pattern'] == 'Weekly'), **tl_models['W13'])
    daily = forecast(target_input.filter(lambda ts: ts.meta['seasonal_pattern'] == 'Daily'), **tl_models['D14'])
    hourly = forecast(target_input.filter(lambda ts: ts.meta['seasonal_pattern'] == 'Hourly'), **tl_models['H48'])
    pd.concat([yearly, quarterly, monthly, weekly, daily, hourly], sort=False).to_csv(
        os.path.join(os.path.join(path, 'M4.csv')))

    # M3
    target_input, _ = M3Dataset(M3Meta.dataset_path).standard_split()
    yearly = forecast(target_input.filter(lambda ts: ts.meta['seasonal_pattern'] == 'M3Year'), **tl_models['Y6'])
    quarterly = forecast(target_input.filter(lambda ts: ts.meta['seasonal_pattern'] == 'M3Quart'), **tl_models['Q8'])
    monthly = forecast(target_input.filter(lambda ts: ts.meta['seasonal_pattern'] == 'M3Month'), **tl_models['M18'])
    others = forecast(target_input.filter(lambda ts: ts.meta['seasonal_pattern'] == 'M3Other'), **tl_models['Q8'])
    pd.concat([yearly, quarterly, monthly, others], sort=False).to_csv(os.path.join(os.path.join(path, 'M3.csv')))

    # Tourism
    target_input, _ = TourismDataset(TourismMeta.dataset_path).standard_split()
    yearly = forecast(target_input.filter(lambda ts: ts.meta['seasonal_pattern'] == 'Yearly'), **tl_models['Y4'])
    quarterly = forecast(target_input.filter(lambda ts: ts.meta['seasonal_pattern'] == 'Quarterly'), **tl_models['Q8'])
    monthly = forecast(target_input.filter(lambda ts: ts.meta['seasonal_pattern'] == 'Monthly'), **tl_models['M24'])
    pd.concat([yearly, quarterly, monthly], sort=False).to_csv(os.path.join(os.path.join(path, 'tourism.csv')))

    # Electricity
    target_input, rolling_target_input = ElectricityDataset(ElectricityMeta.dataset_path). \
        load_cache().split(lambda ts: ts.split(-24 * 7))
    rolling_daily_forecast(base_insample=target_input, rolling_insample=rolling_target_input, **tl_models['H24']). \
        to_csv(os.path.join(os.path.join(path, 'electricity_last_window.csv')))

    target_input, rolling_target_input = ElectricityDataset(ElectricityMeta.dataset_path).load_cache(). \
        split(lambda ts: ts.split_by_time(ElectricityMeta.deepar_split))
    rolling_daily_forecast(base_insample=target_input, rolling_insample=rolling_target_input, **tl_models['H24']). \
        to_csv(os.path.join(os.path.join(path, 'electricity_deepar.csv')))

    target_input, rolling_target_input = ElectricityDataset(ElectricityMeta.dataset_path).load_cache(). \
        split(lambda ts: ts.split_by_time(ElectricityMeta.deepfact_split))
    rolling_daily_forecast(base_insample=target_input, rolling_insample=rolling_target_input, **tl_models['H24']). \
        to_csv(os.path.join(os.path.join(path, 'electricity_deepfactors.csv')))

    # Traffic
    target_input, rolling_target_input = TrafficDataset(TrafficMeta.dataset_path).load_cache().\
        split(lambda ts: ts.split(-24 * 7))
    rolling_daily_forecast(base_insample=target_input, rolling_insample=rolling_target_input, **tl_models['H24']). \
        to_csv(os.path.join(os.path.join(path, 'traffic_last_window.csv')))

    target_input, rolling_target_input = TrafficDataset(TrafficMeta.dataset_path).load_cache(). \
        split(lambda ts: ts.split_by_time(TrafficMeta.deepar_split))
    rolling_daily_forecast(base_insample=target_input, rolling_insample=rolling_target_input, **tl_models['H24']). \
        to_csv(os.path.join(os.path.join(path, 'traffic_deepar.csv')))

    target_input, rolling_target_input = TrafficDataset(TrafficMeta.dataset_path).load_cache(). \
        split(lambda ts: ts.split_by_time(TrafficMeta.deepfact_split))
    rolling_daily_forecast(base_insample=target_input, rolling_insample=rolling_target_input, **tl_models['H24']). \
        to_csv(os.path.join(os.path.join(path, 'traffic_deepfactors.csv')))

    # FRED
    target_input, _ = FredDataset(FredMeta.dataset_path).standard_split()
    yearly = forecast(target_input.filter(lambda ts: ts.meta['seasonal_pattern'] == 'Yearly'), **tl_models['Y6'])
    quarterly = forecast(target_input.filter(lambda ts: ts.meta['seasonal_pattern'] == 'Quarterly'), **tl_models['Q8'])
    monthly = forecast(target_input.filter(lambda ts: ts.meta['seasonal_pattern'] == 'Monthly'), **tl_models['M18'])
    weekly = forecast(target_input.filter(lambda ts: ts.meta['seasonal_pattern'] == 'Weekly'), **tl_models['W13'])
    daily = forecast(target_input.filter(lambda ts: ts.meta['seasonal_pattern'] == 'Daily'), **tl_models['D14'])
    pd.concat([yearly, quarterly, monthly, weekly, daily]).to_csv(os.path.join(os.path.join(path, 'fred.csv')))


def evaluate(name: str, summary_filter: str, validation_mode: bool = False):
    pass


def summary(name: str, summary_filter: str = '*', validation_mode: bool = False):
    pass


if __name__ == '__main__':
    logging.root.setLevel(logging.INFO)
    Fire()
