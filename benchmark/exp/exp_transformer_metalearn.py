import time
import yaml
import sys
sys.path.append('/home/ubuntu/ForecastPFN/academic_comparison/')

import os
import time
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim
from typing import Dict
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from exp.torch_utils import *
from transformer_models.models import FEDformer, Autoformer, Informer, Transformer
from utils.tools import EarlyStopping, TimeBudget, adjust_learning_rate, visual
from utils.metrics import metric

sys.path.append('/home/ubuntu/ForecastPFN/src/')
sys.path.append('/home/ubuntu/ForecastPFN/src/training/')
from training.create_train_test_df import create_train_test_df
import tensorflow as tf


from training.config_variables import Config
from training.constants import PADDING, HISTORY_LEN, TARGET_LEN, TRIM_LEN, TARGET_INDEX
from training.prepare_dataset import filter_unusable_points
from training.utils import load_tf_dataset

warnings.filterwarnings('ignore')


CONTEXT_LENGTH = 500


class Exp_Transformer_Meta(Exp_Basic):
    def __init__(self, args):
        super(Exp_Transformer_Meta, self).__init__(args)
        self.vali_timer = TimeBudget(args.time_budget)
        self.train_timer = TimeBudget(args.time_budget)
        self.test_timer = TimeBudget(args.time_budget)

    def _build_model(self):
        model_dict = {
            'FEDformer': FEDformer,
            'FEDformer_Meta': FEDformer,
            'FEDformer-w': FEDformer,
            'FEDformer-f': FEDformer,
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model.to(self.device)

    def _get_data(self, flag):


        TARGET_LEN = self.args.label_len + self.args.pred_len
        TRIM_LEN = self.args.label_len + self.args.pred_len
        TARGET_INDEX = 2*TRIM_LEN


        def compute_time_features(ts: np.ndarray):
            """
            Method to compute time features to be used by model
            :param ts: array consisting of int64Index representing timestamps
            :return: numpy array of shape (n, 5) containing the time features
            """
            ts = pd.to_datetime(ts)
            if Config.is_sub_day:
                return np.stack([ts.minute, ts.hour, ts.day, ts.day_of_week + 1, ts.day_of_year], axis=-1)
            return np.stack([ts.month, ts.day, ts.day_of_week, ts.hour], axis=-1)


        def build_frames(r: Dict[str, tf.Tensor]):
            raw_date_info = tf.numpy_function(
                compute_time_features, inp=[r['ts']], Tout=tf.int64)
            date_info = tf.signal.frame(
                tf.pad(raw_date_info, [[PADDING, 0], [0, 0]]),
                HISTORY_LEN,
                1,
                axis=0
            )

            history = tf.signal.frame(
                tf.pad(r['y'], [[PADDING, 0]]), HISTORY_LEN, 1, axis=-1)
            noise = tf.signal.frame(
                tf.pad(r['noise'], [[PADDING, 0]]), HISTORY_LEN, 1, axis=-1)

            target_dates = tf.signal.frame(raw_date_info, TARGET_LEN, 1, axis=0)
            target_values = tf.signal.frame(r['y'], TARGET_LEN, 1, axis=-1)
            target_noise = tf.signal.frame(r['noise'], TARGET_LEN, 1, axis=-1)

            start_index = target_values.shape[0] - TRIM_LEN

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

            return dict(
                ts=date_info,
                history=history*noise,
                noise=noise,
                target_ts=target_dates,
                target_noise=target_noise
            ), target_values

        @tf.function
        def gen_random_single_point_no_noise(
            date_info: tf.Tensor,
            history: tf.Tensor,
            noise: tf.Tensor,
            target_dates: tf.Tensor,
            target_values: tf.Tensor,
            target_noise: tf.Tensor
        ):

            return dict(
                ts=date_info,
                history=history,
                noise=noise,
                target_ts=target_dates,
                target_noise=target_noise
            ), target_values


        def remove_noise(x, y):
            return (
                {
                    'ts': x['ts'],
                    'history': x['history'],
                    'target_ts': x['target_ts'],
                }, y
            )

        def create_train_test_df(combined_ds, test_noise=False):
            base_train_df = combined_ds.skip(30).map(build_frames).repeat()
            base_test_df = combined_ds.take(30).map(build_frames)
            task_map = {
                'point': gen_random_single_point,
            }
            train_tasks_dfs = [
                base_train_df.map(func, num_parallel_calls=tf.data.AUTOTUNE)
                for func in task_map.values()
            ]
            train_df = tf.data.Dataset.choose_from_datasets(
                train_tasks_dfs, tf.data.Dataset.range(len(train_tasks_dfs)).repeat()
            ).unbatch().filter(filter_unusable_points)

            task_map_test = {
                'point': gen_random_single_point_no_noise,
            }

            if test_noise:
                test_tasks_dfs = [
                    base_test_df.map(func, num_parallel_calls=tf.data.AUTOTUNE)
                    for func in task_map.values()
                ]
            else:
                test_tasks_dfs = [
                    base_test_df.map(func, num_parallel_calls=tf.data.AUTOTUNE)
                    for func in task_map_test.values()
                ]

            test_df = tf.data.Dataset.choose_from_datasets(
                test_tasks_dfs, tf.data.Dataset.range(len(test_tasks_dfs)).repeat()
            ).unbatch().filter(filter_unusable_points)

            test_df = test_df.map(remove_noise)

            return train_df, test_df


        def get_combined_ds(config):
            version = config["version"]
            datasets = [
                # load_tf_dataset(config["prefix"] + f"{version}/minute.tfrecords"),
                # load_tf_dataset(config["prefix"] + f"{version}/hourly.tfrecords"),
                load_tf_dataset(config["prefix"] + f"{version}/daily.tfrecords"),
                # load_tf_dataset(config["prefix"] + f"{version}/weekly.tfrecords"),
                # load_tf_dataset(config["prefix"] + f"{version}/monthly.tfrecords"),
            ]
            combined_ds = tf.data.Dataset.choose_from_datasets(
                datasets, tf.data.Dataset.range(1).repeat()
            )

            return combined_ds



        if flag == 'test':
            data_set, data_loader = data_provider(self.args, flag)
        elif flag == 'train':
            with open('/home/ubuntu/ForecastPFN/src/training/config_mf_replicate_testnoiseF.yaml') as config_file:
                config = yaml.load(config_file, yaml.loader.SafeLoader)

            combined_ds = get_combined_ds(config)
            train_df, vali_df = create_train_test_df(
                combined_ds, config["test_noise"])
            data_loader = TFRecordDataLoader(
                train_df, self.args.batch_size, True, 10_000)
            data_set = None
        elif flag == 'val':
            with open('/home/ubuntu/ForecastPFN/src/training/config_mf_replicate_testnoiseF.yaml') as config_file:
                config = yaml.load(config_file, yaml.loader.SafeLoader)

            combined_ds = get_combined_ds(config)
            train_df, vali_df = create_train_test_df(
                combined_ds, config["test_noise"])
            data_set = None
            data_loader = TFRecordDataLoader(
                vali_df,  self.args.batch_size, True, 10_000)
        return data_set, data_loader


    def _select_optimizer(self):
        model_optim = optim.Adam(
            self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        self.vali_timer.start_timer()
        with torch.no_grad():
            for batch_i, batch_data in enumerate(tqdm(vali_loader)):
                # Get batch data
                X_batch = numpy_to_torch(batch_data[0], self.device)
                y_batch = torch.from_numpy(batch_data[1]).to(self.device)

                batch_x = X_batch['history'].float().to(
                    self.device).unsqueeze(2)
                batch_y = y_batch.float().to(self.device).unsqueeze(2)

                batch_x_mark = X_batch['ts'].float().to(self.device)
                batch_y_mark = X_batch['target_ts'].float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(
                    batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat(
                    [batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(
                                batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(
                                batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(
                            batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(
                            batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                batch_y = batch_y[:, -self.args.pred_len:,
                                  f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)

        total_loss = np.average(total_loss)
        self.model.train()
        self.vali_timer.end_timer()
        return total_loss

    def test(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        self.vali_timer.start_timer()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(
                    batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat(
                    [batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(
                                batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(
                                batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(
                            batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(
                            batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                batch_y = batch_y[:, -self.args.pred_len:,
                                  f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        self.vali_timer.end_timer()
        return total_loss
    
    def train(self, setting):
        print(setting)

        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            raise NotImplementedError

        time_now = time.time()

        train_steps = -1
        early_stopping = EarlyStopping(
            patience=self.args.patience, verbose=False)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        self.train_timer.start_timer()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for batch_i, batch_data in enumerate(tqdm(train_loader)):
                # Get batch data
                X_batch = numpy_to_torch(batch_data[0], self.device)
                y_batch = torch.from_numpy(batch_data[1]).to(self.device)

                batch_x = X_batch['history'].float().to(self.device).unsqueeze(2)
                batch_y = y_batch.float().to(self.device).unsqueeze(2)
                
                batch_x_mark = X_batch['ts'].float().to(self.device)
                batch_y_mark = X_batch['target_ts'].float().to(self.device)

                iter_count += 1
                model_optim.zero_grad()

                # decoder input
                dec_inp = torch.zeros_like(
                    batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat(
                    [batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(
                                batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(
                                batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        batch_y = batch_y[:, -self.args.pred_len:,
                                          f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(
                            batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(
                            batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    batch_y = batch_y[:, -self.args.pred_len:,
                                      f_dim:].to(self.device)

                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

                self.train_timer.step()
                if self.train_timer.budget_reached:
                    early_stopping.save_checkpoint('', self.model, path)
                    print(f'Budget reached: {self.train_timer.total_time}')
                    self.train_timer.end_timer()

                    best_model_path = path + '/' + 'checkpoint.pth'
                    self.model.load_state_dict(torch.load(best_model_path))

                    return self.model
                
                if batch_i >= 1_000:
                    break

            print("Epoch: {} cost time: {}".format(
                epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.test(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            # if early_stopping.early_stop:
            #     print("Early stopping")
            #     break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        self.train_timer.end_timer()

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

