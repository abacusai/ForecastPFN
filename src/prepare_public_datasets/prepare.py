"""
Module to prepare public datasets for training
"""

import csv
import yaml
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tempfile import NamedTemporaryFile
from reainternal.cloud import CloudLocation
from tqdm import tqdm
from constants import *

def read_timeseries_file(filename):
    """
    Function to read the standard datasets for time series.
    The datasets are in CSV format, hence the function is implemented
    accordingly
    """
    lines = []
    with open(filename) as fh:
        reader = csv.reader(fh)
        for line in reader:
            lines.append([float(x) for x in line])

    return lines

def generate_tf_train_examples(name, train_data_list, freq):
    """
    Method to generate the examples from train data
    tf.train.Example format
    """
    for train_data in train_data_list:
        i = len(train_data)

        while i > 0:
            train_data = train_data[max(i-CONTEXT_LENGTH, 0):i]
            if len(train_data) < CONTEXT_LENGTH:
                train_data = [0] * (CONTEXT_LENGTH - len(train_data)) + train_data

            start = f'{np.random.randint(2010, 2013)}-{np.random.randint(1, 13)}-{np.random.randint(1, 11)}'
            dates = pd.date_range(start=start, periods=CONTEXT_LENGTH, freq=freq)

            noise = [1] * CONTEXT_LENGTH

            i -= WINDOW_STRIDE

            if np.max(train_data) == 0:
                continue

            print(train_data)

            yield tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            "id": tf.train.Feature(bytes_list=tf.train.BytesList(value=[name.encode()])),
                            "ts": tf.train.Feature(
                                int64_list=tf.train.Int64List(
                                    value=dates.astype(np.int64)
                                )
                            ),
                            "y": tf.train.Feature(
                                float_list=tf.train.FloatList(value=train_data)
                            ),
                            "noise": tf.train.Feature(
                                float_list=tf.train.FloatList(value=noise)
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
            tfile.name, options=tf.io.TFRecordOptions(compression_type="GZIP")
        ) as writer:
            for record in tqdm(it):
                writer.write(record.SerializeToString())
        tfile.seek(0)
        CloudLocation(prefix + dest).copy_from_file(tfile)

def save_tf_dataset(prefix: str, dataset_name: str, data: list, freq: str):
    """
    Generate dataset and save as tf records
    """
    save_tf_records(
        prefix,
        f"{dataset_name}.tfrecords",
        generate_tf_train_examples(dataset_name, data, freq)
    )

    print(f"Written to file {dataset_name}.tfrecords")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, help="Path to config file")
    args = parser.parse_args()

    with open(args.config) as config_file:
        config = yaml.load(config_file, yaml.loader.SafeLoader)

    train_data = read_timeseries_file(config['train_path'])
    save_tf_dataset(config['prefix'], config['dataset_name'], train_data, config['freq'])

if __name__ == '__main__':
    main()