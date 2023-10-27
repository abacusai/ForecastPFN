from dataclasses import dataclass
from typing import Callable

import numpy as np

from common.evaluator import Evaluator, EvaluationResult
from common.metrics import nd
from common.timeseries import TimeseriesBundle
from common.utils import round_half_up


@dataclass
class ElectricityEvaluator(Evaluator):
    metric_fn: Callable[[np.ndarray, np.ndarray], float] = nd
    precision: int = 2

    def evaluate(self, forecasts: TimeseriesBundle) -> EvaluationResult:
        return {'metric': round_half_up(self.metric_fn(np.array(forecasts.values()),
                                                       np.array(self.test_set.values())),
                                        self.precision)}
