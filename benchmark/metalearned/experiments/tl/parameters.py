common = {
    'repeats': list(range(10)),

    'lookback_period': list(range(2, 8)),
    'loss_name': 'MASE',
    'scaling': 'max',
    'iterations': 15000,
    'history_horizons': 10,

    'batch_size': 1024,
    'learning_rate': 0.001,

    'mode': 'dress',

    'width': 512,
    'layers': 4,
    'blocks': 10,
    'stacks': 1,

    # interpretable
    'trend_blocks': 3,
    'trend_fc_layers_size': 256,
    'degree_of_polynomial': 3,

    'seasonality_blocks': 3,
    'seasonality_fc_layers_size': 2048,
    'num_of_harmonics': 1,

    'logging_frequency': 500,
    'snapshot_frequency': 5000,
}

parameters = {
    'trel_debug': {
        **common,
        'source_dataset': 'M4',
        'model_type': 'generic',
        'width': 512,
        'blocks': 30,
        'stacks': 1,
        'iterations': [5000, 15000],
        'loss_name': ['MASE', 'MAPE', 'SMAPE']
    },
    'shared_grid': {
        **common,
        'source_dataset': 'M4',
        'model_type': 'generic',
        'width': [512, 1024],
        'stacks': [1],
        'blocks': [1, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50, 75, 100],
        'loss_name': ['MASE', 'MAPE', 'SMAPE'],
    },
    'not_shared_grid': {
        **common,
        'source_dataset': 'M4',
        'model_type': 'generic',
        'width': [512, 1024],
        'stacks': [1, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50, 75, 100],
        'blocks': [1],
        'loss_name': ['MASE', 'MAPE', 'SMAPE'],
    },
    'shared': {
        **common,
        'model_type': 'generic',
        'source_dataset': ['M4', 'FRED'],
        'loss_name': ['MASE', 'MAPE', 'SMAPE'],
        'blocks': 30,
        'stacks': 1,
        'mode': 'dress',
    },
    'not_shared': {
        **common,
        'model_type': 'generic',
        'source_dataset': ['M4', 'FRED'],
        'loss_name': ['MASE', 'MAPE', 'SMAPE'],
        'blocks': 1,
        'stacks': 30,
        'mode': 'dress',
    }
}
