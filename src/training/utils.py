"""
Utility functions for training script
"""
import tensorflow as tf
from constants import TF_SCHEMA


def decode_fn(record_bytes):
    """
    Method to process bytes from tfrecord files
    """
    return tf.io.parse_single_example(record_bytes, TF_SCHEMA)


def load_tf_dataset(src: str):
    """
    Method to load and decode dataset from tfrecord files
    """
    return tf.data.TFRecordDataset(src, compression_type="GZIP").map(decode_fn)
