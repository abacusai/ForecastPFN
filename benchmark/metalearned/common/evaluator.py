from dataclasses import dataclass
from common.timeseries import TimeseriesBundle

@dataclass
class Evaluator:
    test_set: TimeseriesBundle

@dataclass
class EvaluationResult:
    test_set: TimeseriesBundle