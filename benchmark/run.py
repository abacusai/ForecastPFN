import numpy as np
import random
from exp.exp_resolver import resolve_experiment
import torch
import argparse
from utils.arg_resolver import resolve_transformer_args, _model_is_transformer, setting_string, resolve_args

import sys
sys.path.append("metalearned")

def parse():

    parser = argparse.ArgumentParser(
        description='Comparing performance of ForecastPFN to other Time Series Benchmarks')

    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--use_gpu', type=bool, default=True, help='status')
    parser.add_argument('--itr', type=int, default=1, help='status')

    # model settings
    parser.add_argument('--model', type=str, default='ForecastPFN',
                        help='model name, options: [ForecastPFN, FEDformer, Autoformer, Informer, Transformer, Arima, Prophet]')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96,
                        help='input sequence length')
    parser.add_argument('--label_len', type=int,
                        default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96,
                        help='prediction sequence length')

    parser.add_argument('--time_budget', type=int,
                        help='amount of time budget to train the model')
    parser.add_argument('--train_budget', type=int,
                        help='length of training sequence')

    # data loader
    parser.add_argument('--data', type=str,
                        default='ETTh1', help='dataset type')
    parser.add_argument('--root_path', type=str,
                        default='./dataset/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str,
                        default='ETTh1.csv', help='data file')
    parser.add_argument('--target', type=str,
                        default='OT', help='name of target column')
    parser.add_argument('--scale', type=bool, default=True,
                        help='scale the time series with sklearn.StandardScale()')

    # ForecastPFN
    parser.add_argument('--model_path', type=str, default='s3://realityengines.datasets/forecasting/pretrained/gurnoor/models/20230202-025828/ckpts',
                        help='encoder input size')
    parser.add_argument('--scaler', type=str, default='standard',
                        help='scale the test series with sklearn.StandardScale()')

    # Metalearn
    parser.add_argument('--metalearn_freq', type=str,
                        help='which type of model should be used for the Metalearn model. Typically M, W, or D.')
    return parser


def main():
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = parse()

    args = parser.parse_args()

    args = resolve_args(args)
    if _model_is_transformer(args.model):
        args = resolve_transformer_args(args)
    
    if args.model != 'ForecastPFN':
        args.model_name = None
    else:
        args.model_name = args.model_path.split('/')[-2]

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.dvices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]


    print('Args in experiment:')
    print(args)

    exp = resolve_experiment(args)

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            setting = setting_string(args, ii)

            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)

            torch.cuda.empty_cache()
            exp.reset()
    else:
        ii = 0
        setting = setting_string(args, ii)
        

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
