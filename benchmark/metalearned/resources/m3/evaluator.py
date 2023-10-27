from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

from common.evaluator import Evaluator, EvaluationResult
from common.metrics import smape_1, smape_2
from common.timeseries import TimeseriesBundle
from resources.m3.dataset import M3Meta


@dataclass
class M3Evaluator(Evaluator):
    smape_1: bool = True

    def evaluate(self, forecasts: TimeseriesBundle) -> EvaluationResult:
        results = OrderedDict()
        cumulative_metrics = 0
        cumulative_points = 0
        offset = 0

        evaluation_function = smape_1 if self.smape_1 else smape_2

        for sp in M3Meta.seasonal_patterns:
            target_sp = self.test_set.filter(lambda ts: ts.meta['seasonal_pattern'] == sp)
            forecast_sp = forecasts.filter(lambda ts: ts.meta['seasonal_pattern'] == sp)

            target, forecast = target_sp.intersection_by_id(forecast_sp)

            target_values = np.array(target.values())
            forecast_values = np.array(forecast.values())

            assert target_values.shape == forecast_values.shape

            metric = evaluation_function(target_values, forecast_values)
            cumulative_metrics += np.sum(metric)
            cumulative_points += np.prod(target_values.shape)
            results[sp] = round(float(np.mean(metric)), 2)
            offset += len(target_values)

        results['Average'] = round(cumulative_metrics / cumulative_points, 2)
        return results
