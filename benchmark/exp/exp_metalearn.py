import os
import warnings
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import datetime
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.metrics import metric
from utils.metrics import smape
import tensorflow as tf
import tensorflow_io
import time
from sklearn.preprocessing import StandardScaler, MinMaxScaler
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120)])
  except RuntimeError as e:
    print(e)

import sys
sys.path.append('metalearned')

from metalearned.common.experiment import load_experiment_parameters
from metalearned.common.torch_utils import SnapshotManager, to_device, to_tensor, mase_loss, mape_loss, smape_2_loss
from metalearned.models.nbeats_torch import nbeats_generic, nbeats_interpretable




warnings.filterwarnings('ignore')


class Exp_Metalearn(Exp_Basic):
    def __init__(self, args):
        super(Exp_Metalearn, self).__init__(args)

    def _build_model(self):

        self.args.path = f'metalearned/experiments/tl/ForecastPFN/loss_name=MAPE,input_size={self.args.seq_len},horizon={self.args.pred_len}/'

        experiment_parameters = load_experiment_parameters(self.args.path)
        self.args.experiment_parameters = experiment_parameters

        input_size = experiment_parameters['input_size']
        horizon = experiment_parameters['horizon']

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
                                        seasonality_fc_layers_size=experiment_parameters[
                                            'seasonality_fc_layers_size'],
                                        num_of_harmonics=experiment_parameters['num_of_harmonics'],
                                        scaling=experiment_parameters['scaling'],
                                        mode=experiment_parameters['mode'])

        return model.to(self.device)

    def train(self, setting):
        return
    
    
    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')

        if test:
            print('loading model')

            time_freq = self.args.metalearn_freq

            path = self.args.path
            experiment_parameters = self.args.experiment_parameters

            snapshot_dir = os.path.join(path, 'snapshots', time_freq)
            snapshot_manager = SnapshotManager(snapshot_dir=snapshot_dir,
                                            logging_frequency=experiment_parameters['logging_frequency'],
                                            snapshot_frequency=experiment_parameters['snapshot_frequency'])

            self.model.load_state_dict(torch.load(snapshot_manager.model_snapshot_file))
            self.model.to(self.device)


        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.test_timer.start_timer()
        timer = 0
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)[:,:,0]
                batch_y = batch_y.float().to(self.device)[:,:,0]

                print(batch_x.shape, batch_y.shape)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                t1 = time.time()
                pred = self.model(batch_x, torch.ones(batch_x.shape).to(self.device))
                timer += time.time()-t1

                pred = pred.detach().cpu().numpy()
                true = batch_y[:,-self.args.pred_len:].detach().cpu().numpy()

                preds.append(pred)
                trues.append(true)

        self.test_timer.end_timer()
        self.test_timer.total_time = timer
        print('timer: ', timer)

        return self._save_test_data(setting, preds, trues)
