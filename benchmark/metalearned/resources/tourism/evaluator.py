from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable

import numpy as np

from common.evaluator import Evaluator, EvaluationResult
from common.metrics import mape
from common.timeseries import TimeseriesBundle
from resources.tourism.dataset import TourismMeta


@dataclass
class TourismEvaluator(Evaluator):
    metric_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] = mape
    precision: int = 2

    def evaluate(self, forecasts: TimeseriesBundle) -> EvaluationResult:
        results = OrderedDict()
        cumulative_metrics = 0
        cumulative_points = 0
        offset = 0

        for sp in TourismMeta.seasonal_patterns:
            target_for_sp = self.test_set.filter(lambda ts: ts.meta['seasonal_pattern'] == sp)
            forecast_for_sp = forecasts.filter(lambda ts: ts.meta['seasonal_pattern'] == sp)

            target = np.array(target_for_sp.values())
            forecast = np.array(forecast_for_sp.values())

            # This is avoid problems on the validation set, because there are some zeros in the Monthly subset
            zeros_indicator = target == 0
            target[zeros_indicator] = 1.0
            forecast[zeros_indicator] = 1.0

            score = self.metric_fn(forecast, target)
            cumulative_metrics += np.sum(score)
            cumulative_points += np.prod(target.shape)
            results[sp] = round(float(np.mean(score)), self.precision)
            offset += len(target)

        results['Average'] = round(cumulative_metrics / cumulative_points, self.precision)
        return results
