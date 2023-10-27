import os
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.metrics import metric
import pmdarima
import pandas as pd

warnings.filterwarnings('ignore')


class Exp_Arima(Exp_Basic):
    def __init__(self, args):
        super(Exp_Arima, self).__init__(args)

    def _build_model(self):
        return pmdarima.auto_arima


    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        train_df = pd.DataFrame({'y': train_data.data_y.T[0], 'ds': list(
            pd.to_datetime(train_data.data_stamp_original['date']))})
        self.train_timer.start_timer()
        self.model = pmdarima.auto_arima(train_df.y.values)
        self.train_timer.end_timer()
        return

    def test(self, setting, test=0):
        horizon = self.args.pred_len

        test_data, test_loader = self._get_data(flag='test')
        test_df = pd.DataFrame({'y': test_data.data_y.T[0], 'ds': list(
            pd.to_datetime(test_data.data_stamp_original['date']))})

        cmp = pd.DataFrame({
            'date': test_df['ds'].values,
            'y': test_df['y'].values,
            'yhat': self.model.predict(test_df.shape[0])
        })

        preds, trues = [], []
        self.test_timer.start_timer()
        for i in range(self.args.seq_len, cmp.shape[0]-horizon+1):
            pred = cmp[i:i+horizon]['yhat'].values
            true = cmp[i:i+horizon]['y'].values
            preds += [pred]
            trues += [true]

        self.test_timer.end_timer()
        
        return self._save_test_data(setting, preds, trues)
