import os
import torch
import numpy as np
from data_provider.data_factory import data_provider
from utils.tools import TimeBudget
from utils.metrics import metric

class Exp_Basic(object):
    def __init__(self, args):
        if args is not None:
            self.args = args
            self.device = self._acquire_device()
            self.model = self._build_model()
            self.vali_timer = TimeBudget(args.time_budget)
            self.train_timer = TimeBudget(args.time_budget)
            self.test_timer = TimeBudget(args.time_budget)

    def _build_model(self):
        raise NotImplementedError

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _save_test_data(self, setting, preds, trues):
        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        output = {
            'metrics': {
                'mae': mae,
                'mse': mse, 
                'rmse': rmse, 
                'mape': mape, 
                'mspe': mspe,
            },
            'train_timer': self.train_timer.total_time,
            'vali_timer': self.vali_timer.total_time,
            'test_timer': self.test_timer.total_time,
            'args': self.args
        }
        print(output)

        np.save(folder_path + 'metrics.npy', output)
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return mae, mse, rmse, mape, mspe

    def vali(self, *args, **kwargs):
        pass

    def train(self, *args, **kwargs):
        pass

    def test(self, *args, **kwargs):
        pass

    def reset(self, *args, **kwargs):
        pass
