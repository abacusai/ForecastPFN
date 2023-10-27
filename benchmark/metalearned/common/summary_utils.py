import itertools
import logging
import random
from glob import glob
from typing import List

import pandas as pd
from tqdm import tqdm


class EnsembleStatistics:
    def __init__(self, filter_path: str, evaluator):
        """
        Class for building ensembles statistics.

        Example:

        ```
        hp_summary = HyperParametersSummary('/project/experiments/nbeats/m3/ad_search_30_val/**/forecast.csv',
                                             evaluator=M3Summary(validation_mode=True, use_smape2=True))
        statistics = hp_summary.bootstrap(bootstrap_key='repeat',
                                          bootstrap_size=5,
                                          group_keys=['history_size', 'iterations'],
                                          number_of_samples=100)
        ```

        :param filter_path: Pattern to find CSV files with forecasts.
        :param evaluator: Class which provides `evaluate` method, which takes pandas dataframe and returns
                          dictionary of group -> performance (float)
        """
        self.predictions = []
        self.parameters = []
        self.evaluator = evaluator
        self.groups = {}
        for f in tqdm(glob(filter_path)):
            self.predictions.append(pd.read_csv(f).set_index('id'))
            parameters = f.split('/')[-2]
            self.parameters.append(parameters)
            for parameter in parameters.split(','):
                parameter_key, parameter_value = parameter.split('=')
                if parameter_key not in self.groups:
                    self.groups[parameter_key] = {}
                if parameter_value not in self.groups[parameter_key]:
                    self.groups[parameter_key][parameter_value] = []
                self.groups[parameter_key][parameter_value].append(len(self.predictions) - 1)
        self.group_names = ', '.join(self.groups.keys())
        logging.debug(f'Loaded {len(self.predictions)} predictions')
        logging.debug(f'Parameters: {self.group_names}')

    def bootstrap(self,
                  ensemble_keys: List[str],
                  bootstrap_key: str,
                  bootstrap_size: int,
                  number_of_samples: int):
        group_keys = self.groups.keys() - set(ensemble_keys)
        group_values = list(itertools.product(*map(lambda g: self.groups[g].keys(), group_keys)))

        results = []
        for group_instance in tqdm(group_values):
            group_ids = [set(self.groups[group_key][group_value]) for group_key, group_value in
                         list(zip(group_keys, group_instance))]
            group_filter = set.intersection(*group_ids) if len(group_ids) > 0 else None
            if group_instance != () and (group_filter is None or len(group_filter) == 0):
                continue
            for _ in range(number_of_samples):
                sampled_ids = set(
                    itertools.chain(*random.sample(list(self.groups[bootstrap_key].values()), k=bootstrap_size)))
                ensemble_ids = sampled_ids.intersection(group_filter) if group_filter is not None else sampled_ids
                if ensemble_ids is None or len(ensemble_ids) == 0:
                    continue
                ensemble_predictions = pd.concat([self.predictions[i]
                                                  for i in ensemble_ids],
                                                 sort=False).groupby(level='id', sort=False).median()
                group_columns = dict(zip(group_keys, group_instance))
                evaluation_results = self.evaluator.evaluate(ensemble_predictions.values)
                for evaluation_key, evaluation_value in evaluation_results.items():
                    results.append(pd.DataFrame({
                        'metric': evaluation_value,
                        'evaluation_key': evaluation_key,
                        **group_columns}, index=[0]))
        return pd.concat(results, sort=False).reset_index()
