import inspect
import logging
import os
import pathlib
import sys
import urllib
from decimal import Decimal, ROUND_HALF_UP
from glob import glob
from itertools import dropwhile, takewhile
from typing import Any, Callable, List
from urllib import request

import numpy as np
import pandas as pd
from math import pow
from tqdm import tqdm


def get_module_path():
    module_path = os.path.dirname(inspect.stack()[1].filename)
    if module_path.startswith('/project/source/'):  # happens in jupyterlab
        module_path = module_path.replace('/project/source/', '')
    return module_path


def round_half_up(n, precision):
    return int(Decimal(n * pow(10, precision)).to_integral_value(rounding=ROUND_HALF_UP)) / pow(10, precision)


def median_ensemble(experiment_path: str,
                    summary_filter: str = '**',
                    forecast_file: str = 'forecast.csv',
                    group_by: str = 'id'):
    return pd.concat([pd.read_csv(file)
                      for file in
                      tqdm(glob(os.path.join(experiment_path, summary_filter, forecast_file)))], sort=False) \
        .set_index(group_by).groupby(level=group_by, sort=False).median().values


def group_values(values: np.ndarray, groups: np.ndarray, group_name: str):
    return np.array([v[~np.isnan(v)] for v in values[groups == group_name]])


def download_url(url: str, file_path: str) -> None:
    """
    Download a file to the given target directory.

    :param url: URL to download
    :param file_path: Where to download content.
    """

    def progress(count, block_size, total_size):
        sys.stdout.write('\rDownloading {} from {} {:.1f}%'.format(file_path, url, float(count * block_size) / float(
            total_size) * 100.0))
        sys.stdout.flush()

    if not os.path.isfile(file_path):
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-agent', 'Mozilla/5.0')]
        urllib.request.install_opener(opener)
        pathlib.Path(os.path.dirname(file_path)).mkdir(parents=True, exist_ok=True)
        f, _ = request.urlretrieve(url, file_path, progress)
        sys.stdout.write('\n')
        sys.stdout.flush()
        file_info = os.stat(f)
        logging.info(f'Successfully downloaded {os.path.basename(file_path)} {file_info.st_size} bytes.')
    else:
        file_info = os.stat(file_path)
        logging.info(f'File already exists: {file_path} {file_info.st_size} bytes.')


def url_file_name(url: str) -> str:
    """
    Extract file name from url (last part of the url).
    :param url: URL to extract file name from.
    :return: File name.
    """
    return url.split('/')[-1] if len(url) > 0 else ''


def clean_nans(values):
    return np.array([v[~np.isnan(v)] for v in values])


def ordered_insert(ordered_stack: List, value, f: Callable[[Any, Any], bool]):
    """
    Insert element in a stack by copying first elements of the given stack if they satisfy the condition defined by f,
    then inserts the given value then adds the rest of the values of the stack. In the end it preserves the initial
    stack size, by truncating the result.

    For example, to keep top 3 results with maximum score this method can be used as:
    [2] ordered_insert([-np.inf, -np.inf, -np.inf], 0.3, lambda s, v: s > v)
    > [0.3, -inf, -inf]
    [3] ordered_insert([0.3, -np.inf, -np.inf], 0.8, lambda s, v: s > v)
    > [0.8, 0.3, -inf]
    [4] ordered_insert([0.8, 0.3, -np.inf], 0.9, lambda s, v: s > v)
    > [0.9, 0.8, 0.3]
    [5] ordered_insert([0.9, 0.8, 0.3], 0.99, lambda s, v: s > v)
    > [0.99, 0.9, 0.8]
    [6] ordered_insert([0.99, 0.9, 0.8], 0.5, lambda s, v: s > v)
    > [0.99, 0.9, 0.8]

    :param ordered_stack: Ordered stack so that the v will be inserted at first satisfied condition defined by f.
    :param value: Value to insert, if the value does not satisfy condition f it will not be inserted.
    :param f: Function which takes stack element and the given value to determine if the given value should be inserted
    in the place of the current element, the current element and all elements alter will be pushed down
    (and truncated if necessary).
    :return: New instance of stack with inserted element.
    """
    return (list(takewhile(lambda x: f(x, value), ordered_stack)) + [value] +
            list(dropwhile(lambda x: f(x, value), ordered_stack)))[:len(ordered_stack)]
