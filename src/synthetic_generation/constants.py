"""
Module containing constants for synthetic data generation
"""

from datetime import date

import tensorflow as tf

BASE_START = date.fromisoformat('1885-01-01').toordinal()
BASE_END = date.fromisoformat('2023-12-31').toordinal() + 1

PRODUCT_SCHEMA = {
    'doc': 'Timeseries sample',
    'name': 'TimeseriesSample',
    'type': 'record',
    'fields': [
        {'name': 'id', 'type': 'string'},
        {'name': 'ts', 'type': {'type': 'int', 'logicalType': 'date'}},
        {'name': 'y', 'type': ['null', 'float']},
        {'name': 'noise', 'type': ['float']},
    ],
}

CONTEXT_LENGTH = 1_000

TF_SCHEMA = {
    'id': tf.io.FixedLenFeature([], dtype=tf.string),
    'ts': tf.io.FixedLenFeature([CONTEXT_LENGTH], dtype=tf.int64),
    'y': tf.io.FixedLenFeature([CONTEXT_LENGTH], dtype=tf.float32),
    'noise': tf.io.FixedLenFeature([CONTEXT_LENGTH], dtype=tf.float32),
}
