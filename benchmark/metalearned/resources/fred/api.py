import datetime
import logging
import os
import pickle
import tempfile
import time
from typing import Callable, TypeVar

import numpy as np
from fred import Fred
from tqdm import tqdm

"""
Hourly aggregated dataset from https://archive.ics.uci.edu/ml/datasets/PEMS-SF

As it is used in https://www.cs.utexas.edu/~rofuyu/papers/tr-mf-nips.pdf
Dataset was also compared with the one built by the TRMF paper's author:
https://github.com/rofuyu/exp-trmf-nips16/blob/master/python/exp-scripts/datasets/download-data.sh
"""

A = TypeVar('A')


class FredAPI:
    def __init__(self, dataset_path):
        # request an api key from here: https://research.stlouisfed.org/docs/api/api_key.html
        key_path = 'key.txt'
        self.max_retries = 20
        self.wait_delay = 20.0
        self.dataset_path = dataset_path

        if not os.path.isfile(key_path):
            raise Exception(f'Cannot find FRED key file. Create an API key and place it in {key_path}. '
                            'https://research.stlouisfed.org/docs/api/api_key.html')

        with open(key_path, 'r') as f:
            key = f.readline().strip()
        self.api = Fred(api_key=key, response_type='df')

    def call(self, api_fn: Callable[[Fred], A], attempt=1) -> A:
        if attempt > self.max_retries:
            raise Exception('Maximum retries exceeded')
        try:
            return api_fn(self.api)
        except Exception as e:
            # logging.info(f'API Error: {str(e)}. Waiting {self.wait_delay} seconds to retry. Attempt: {attempt}')
            time.sleep(self.wait_delay)
            return self.call(api_fn=api_fn, attempt=attempt + 1)

    def fetch_categories(self, parent=0):
        children = self.call(lambda api: api.category.children(parent))
        subtree = [parent]
        if len(children) > 0:
            for child_id in list(children.id):
                subtree += self.fetch_categories(child_id)
        return subtree

    def fetch_observation(self, timeseries_id: str):
        try:
            values = self.api.series.observations(timeseries_id, params={'output_type': 1,
                                                                         'realtime_start': '1776-07-04'})
            values = values.groupby('date').head(1)
            values = values.set_index('date')['value']
        except Exception as e:
            if 'The series does not exist in ALFRED but may exist in FRED' in str(e) \
                    or 'this exceeds the maximum number of vintage dates allowed' in str(e).lower() \
                    or 'bad request' in str(e).lower():
                # There are a couple of situations where ALFRED (vintage data)
                # would not work properly
                values = self.api.series.observations(timeseries_id)
                values = values.set_index('date')['value']
            elif 'out of bounds nanosecond timestamp' in str(e).lower():
                # Some series like HPGDPUKA (GDP in the UK) start before 1600
                # which does not seem to be supported by Pandas. Return an empty
                # DataFrame for these (`None` is not supported by HDF5).
                return np.empty(0)
            else:
                raise e
        return values.values

    def fetch_all(self):
        if os.path.isfile(os.path.join(self.dataset_path, '_SUCCESS')):
            return

        categories_cache_path = os.path.join(self.dataset_path, 'categories.pickle')
        if os.path.isfile(categories_cache_path):
            with open(categories_cache_path, 'rb') as f:
                categories = pickle.load(f)
                logging.info(f'Loaded {len(categories)} categories')
        else:
            logging.info(f'Fetching categories')
            categories = self.fetch_categories()
            logging.info(f'Fetched {len(categories)} categories')
            with open(categories_cache_path, 'wb') as f:
                pickle.dump(categories, f, protocol=pickle.HIGHEST_PROTOCOL)

        #
        # Fetch timeseries
        #
        logging.info(f'Fetching timeseries')
        dataset_file_path = os.path.join(self.dataset_path, 'dataset.pickle')

        dataset = {'processed_categories': [], 'data': {}}
        if os.path.exists(dataset_file_path):
            with open(dataset_file_path, 'rb') as cache_file_name:
                dataset = pickle.load(cache_file_name)

        categories_to_process = [c for c in categories if c not in dataset['processed_categories']]

        limit = 1000
        for category_id in tqdm(categories_to_process):
            offset = 0
            while True:
                timeseries_meta = self.call(lambda api: api.category.series(category_id, params={'limit': limit,
                                                                                                 'offset': offset}))
                if len(timeseries_meta) == 0:
                    break

                for _, ts_meta in timeseries_meta.iterrows():
                    ts_id = str(ts_meta.id)
                    start_date = datetime.datetime.strptime(str(ts_meta.observation_start), '%Y-%m-%d %H:%M:%S')
                    time_unit = str(ts_meta.frequency)
                    if ts_id not in dataset['data']:
                        dataset['data'][ts_id] = {
                            'start_date': start_date,
                            'time_unit': time_unit,
                            'meta': {
                                'categories': [category_id]
                            },
                            'values': self.call(lambda api: self.fetch_observation(ts_id))
                        }
                    else:
                        dataset['data'][ts_id]['meta']['categories'].append(category_id)
                offset += 1

            dataset['processed_categories'].append(category_id)
            temp_file = tempfile.NamedTemporaryFile(dir=self.dataset_path, delete=False, mode='wb')
            pickle.dump(dataset, temp_file, protocol=pickle.HIGHEST_PROTOCOL)
            temp_file.flush()
            os.fsync(temp_file.fileno())
            os.rename(temp_file.name, dataset_file_path)

