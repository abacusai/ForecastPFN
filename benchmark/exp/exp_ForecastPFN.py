import os
import warnings
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import datetime
import time
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.metrics import metric
from utils.metrics import smape
import tensorflow as tf
import tensorflow_io
from sklearn.preprocessing import StandardScaler, MinMaxScaler
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120)])
  except RuntimeError as e:
    print(e)


warnings.filterwarnings('ignore')


class Exp_ForecastPFN(Exp_Basic):
    def __init__(self, args):
        super(Exp_ForecastPFN, self).__init__(args)

    def _build_model(self):
        return

    def train(self, setting):
        return 
    
    def _ForecastPFN_time_features(self, ts: np.ndarray):
        if type(ts[0]) == datetime.datetime:
            year = [x.year for x in ts]
            month = [x.month for x in ts]
            day = [x.day for x in ts]
            day_of_week = [x.weekday()+1 for x in ts]
            day_of_year = [x.timetuple().tm_yday for x in ts]
            return np.stack([year, month, day, day_of_week, day_of_year], axis=-1)
        ts = pd.to_datetime(ts)
        return np.stack([ts.year, ts.month, ts.day, ts.day_of_week + 1, ts.day_of_year], axis=-1)

    def _process_tuple(self,x,x_mark,y_mark, 
                       model, horizon):
        """
        x: tensor of shape (n, 1)
        x_mark: tensor of shape (n, d)
        y_mark: tensor of shape (horizon, d)

        where
        n       is the input  sequence length
        horizon is the output sequence length
        d is the dimensionality of the time_stamp (5 for ForecastPFN)
        """
        # values of history
        # values may all be the same, in which case we have to adjust last value
        if torch.all(x == x[0]):
            x[-1] += 1
        history = x.cpu()
        scaler = StandardScaler()
        scaler.fit(history.cpu())
        history = scaler.transform(history.cpu())

        # mean of history's last 6 values
        history_mean = np.nanmean(history[-6:])

        # std of history's last 6 values
        history_std = np.nanstd(history[-6:])

        # local scale, don't know why defined so
        local_scale = (history_mean + history_std + 1e-4)

        # change history based on local scale, to normalize it between 0 and 1
        history = np.clip(history / local_scale, a_min=0, a_max=1)

        if x.shape[0] != 100:
            x_mark = tf.convert_to_tensor(x_mark.cpu(), dtype=tf.int64)
            if x.shape[0] > 100:
                target = tf.convert_to_tensor(x_mark)[-100:, :]
                history = tf.convert_to_tensor(history)[-100:, :]
            else:
                target = tf.pad(x_mark.cpu(), [[100-x.shape[0], 0], [0, 0]])
                history = tf.pad(history, [[100-x.shape[0], 0], [0, 0]])

            history = tf.repeat(tf.expand_dims(history, axis=0), [
                                horizon], axis=0)[:, :, 0]
            ts = tf.repeat(tf.expand_dims(
                target, axis=0), [horizon], axis=0)

        else:
            ts = tf.convert_to_tensor(x_mark.unsqueeze(0).repeat(
                horizon, 1, 1), dtype=tf.int64)
            history = tf.convert_to_tensor(history, dtype=tf.float32)

        task = tf.fill([horizon, ], 1)
        target_ts = tf.convert_to_tensor(
            y_mark.cpu()[-horizon:, :].unsqueeze(1), dtype=tf.int64)

        model_input = {'ts': ts, 'history': history,
                        'target_ts': target_ts, 'task': task}
        t1 = time.time()
        pred_vals = model(model_input)
        time_diff = time.time() - t1
        scaled_vals = pred_vals['result'].numpy(
        ).T.reshape(-1) * pred_vals['scale'].numpy().reshape(-1)
        scaled_vals = scaler.inverse_transform([scaled_vals])
        return scaled_vals, time_diff
    
    def _ForecastPFN_process_batch(self, model, batch_x, batch_y, batch_x_mark, batch_y_mark):
        preds = []
        trues = []
        for idx, (x, y, x_mark, y_mark) in enumerate(zip(batch_x, batch_y, batch_x_mark, batch_y_mark)):

            pred, time_diff = self._process_tuple(
                x, x_mark, y_mark, model, self.args.pred_len)

            y = y[-self.args.pred_len:, :].to(self.device)
            true = y.detach().cpu().numpy()
            
            preds += [pred]
            trues += [true]
        return preds, trues, time_diff

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        test_data.data_stamp = self._ForecastPFN_time_features(
            list(test_data.data_stamp_original['date']))
        if test:
            print('loading model')
            pretrained = tf.keras.models.load_model(
                self.args.model_path, custom_objects={'smape': smape})

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.test_timer.start_timer()
        timer = 0
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                pred, true, time = self._ForecastPFN_process_batch(
                    pretrained, batch_x, batch_y, batch_x_mark, batch_y_mark)
                timer += time
                
                preds.append(pred)
                trues.append(true)

        self.test_timer.end_timer()
        self.test_timer.total_time = timer
        print('total time:')
        print(timer)

        return self._save_test_data(setting, preds, trues)
