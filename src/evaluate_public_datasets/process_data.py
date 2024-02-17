"""
Module to process the real world datasets and store
them as a tfrecords file
"""
import csv

import pandas as pd


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


def generate_tf_test_examples(name, train_data, test_data, freq):
    len_data = len(train_data) + len(test_data)
    dates = pd.date_range(start='2010-01-01', periods=len_data, freq=freq)

    return name, dates, train_data, test_data
