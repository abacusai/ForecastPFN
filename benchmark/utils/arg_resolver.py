from sklearn.preprocessing import StandardScaler, MinMaxScaler

def _model_is_transformer(model):
    if model in ['FEDformer', 'FEDformer-f', 'FEDformer-w', 'FEDformer_Meta', 'Autoformer', 'Informer', 'Transformer']:
        return True
    return False

def setting_string(args, ii):
    setting = '{}_{}_sl{}_ll{}_pl{}_timebudget_{}_trainbudget_{}_model-path_{}_itr_{}'.format(
        args.model,
        args.data,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.time_budget,
        args.train_budget,
        args.model_name,
        ii)
    return setting


def resolve_args(args):
    args.features = 'S'
    args.freq = 'h'
    args.checkpoints = './checkpoints/'
    args.embed = 'timeF'
    args.batch_size = 32
    args.gpu = 0
    args.use_multi_gpu = False
    args.devices = '0,1'
    args.num_workers = 10
    if args.scaler == 'standard':
        args.scaler = StandardScaler()
    if args.scaler == 'minmax':
        args.scaler = MinMaxScaler()
    return args



def resolve_transformer_args(args):
    args.mode_select = 'random'
    args.modes = 64
    args.L = 3
    args.base = 'legendre'
    args.cross_activation = 'tanh'

    args.enc_in = 1
    args.dec_in = 1
    args.c_out = 1
    args.d_model = 512
    args.n_heads = 8
    args.e_layers = 2
    args.d_layers = 1
    args.d_ff = 2048
    args.moving_avg = [24]
    args.factor = 3
    args.distil = True
    args.dropout = 0.05
    args.activation = 'gelu'
    args.output_attention = False
    args.do_predict = False
    args.train_epochs = 10
    args.patience = 3
    args.learning_rate = 0.0001
    args.des = 'Exp'
    args.loss = 'mse'
    args.lradj = 'type1'
    args.use_amp = False

    if args.model == 'FEDformer-w':
        args.version = 'Wavelet'
    elif args.model == 'FEDformer-f':
        args.version = 'Fourrier'
    elif 'FEDformer' in args.model:
        args.version = 'Wavelet'

    return args
