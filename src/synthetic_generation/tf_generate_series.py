"""
Module to convert process synthetic series using tensorflow
"""

from datetime import date
from tempfile import NamedTemporaryFile

import fastavro
import numpy as np
import pandas as pd
import tensorflow as tf
from constants import CONTEXT_LENGTH
from generate_series import generate
from reainternal.cloud import CloudLocation
from series_config import PRODUCT_SCHEMA, TF_SCHEMA


def tf_generate_n(
    N=100,
    size=CONTEXT_LENGTH,
    freq_index: int = None,
    start: pd.Timestamp = None,
    options: dict = {},
):
    """
    Generate time series as tf.train.Example
    """

    for i in range(N):
        if i % 1000 == 0:
            print(f'Completed: {i}')

        if i < N * options.get('linear_random_walk_frac', 0):
            cfg, sample = generate(
                size,
                freq_index=freq_index,
                start=start,
                options=options,
                random_walk=True,
            )
        else:
            cfg, sample = generate(
                size, freq_index=freq_index, start=start, options=options
            )
        # cfg is the name of the time series
        # sample is a pandas dataframe where
        #   the index is the datetime object
        #   columns `series_value` and `noise`

        id_ = str(cfg).encode()
        yield tf.train.Example(
            features=tf.train.Features(
                feature={
                    'id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[id_])),
                    'ts': tf.train.Feature(
                        int64_list=tf.train.Int64List(
                            value=sample.index.astype(np.int64)
                        )
                    ),
                    'y': tf.train.Feature(
                        float_list=tf.train.FloatList(value=sample.series_values.values)
                    ),
                    'noise': tf.train.Feature(
                        float_list=tf.train.FloatList(value=sample.noise.values)
                    ),
                }
            )
        )


def save_tf_records(prefix: str, dest: str, it):
    """
    Save tf records on cloud location
    """
    with NamedTemporaryFile() as tfile:
        with tf.io.TFRecordWriter(
            tfile.name, options=tf.io.TFRecordOptions(compression_type='GZIP')
        ) as writer:
            for record in it:
                writer.write(record.SerializeToString())
        tfile.seek(0)
        CloudLocation(prefix + dest).copy_from_file(tfile)


def decode_fn(record_bytes):
    return tf.io.parse_single_example(record_bytes, TF_SCHEMA)


def load_tf_dataset(prefix: str, src: str):
    return tf.data.TFRecordDataset(prefix + src, compression_type='GZIP').map(decode_fn)


def convert_tf_to_rows(records):
    for i, r in enumerate(records):
        if i % 1000 == 0:
            print(f'Completed: {i}')
        id_ = r['id'].decode()
        for ts, y, noise in zip(
            (date.fromtimestamp(v / 1_000_000_000) for v in r['ts']),
            (float(v) for v in r['y']),
            (float(_noise) for _noise in r['noise']),
        ):
            yield {'id': id_, 'ts': ts, 'y': y, 'noise': noise}


def generate_product_input(prefix: str, dest: str, it):
    """
    Write generated dataset into avro files
    """
    with CloudLocation(prefix + dest).open(mode='wb') as file:
        fastavro.writer(file, PRODUCT_SCHEMA, it, codec='deflate')
