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


class Exp_Mean(Exp_Basic):
    def __init__(self, args):
        super(Exp_Mean, self).__init__(args)

    def _build_model(self):
        return pmdarima.auto_arima


    def train(self, setting):
        return

    def test(self, setting, test=0):
        horizon = self.args.pred_len

        test_data, test_loader = self._get_data(flag='test')

        preds, trues = [], []
        self.test_timer.start_timer()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                true = batch_y[:, -self.args.pred_len:].detach().cpu().numpy()
                pred = batch_x.mean(1).unsqueeze(1).repeat(
                    1, true.shape[1], 1).detach().cpu().numpy()

                preds.append(pred)
                trues.append(true)
        self.test_timer.end_timer()

        return self._save_test_data(setting, preds, trues)
