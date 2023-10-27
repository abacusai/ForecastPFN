from collections import OrderedDict
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

from common.evaluator import Evaluator, EvaluationResult
from common.metrics import smape_2
from common.timeseries import TimeseriesBundle
from common.utils import round_half_up
from resources.fred.dataset import FredDataset, FredMeta


@dataclass
class FredEvaluator(Evaluator):
    validation: bool = False

    def evaluate(self, forecast: TimeseriesBundle) -> EvaluationResult:
        insamples, _ = FredDataset(FredMeta.dataset_path).standard_split()
        if self.validation:
            horizons_map = FredMeta().horizons_map()
            insamples, _ = insamples.split(lambda ts: ts.split(-horizons_map[ts.meta['seasonal_pattern']]))

        grouped_smapes = {sp: np.mean(smape_2(forecast=np.array(self.filter_by_sp(forecast, sp).values()),
                                              target=np.array(self.filter_by_sp(self.test_set, sp).values())))
                          for sp in FredMeta.seasonal_patterns}

        grouped_smapes = self.summarize_groups(grouped_smapes)

        return self.round_values(grouped_smapes)

    def summarize_groups(self, scores):
        scores_summary = OrderedDict()

        weighted_score = {}
        for sp in ['Yearly', 'Quarterly', 'Monthly', 'Weekly', 'Daily']:
            weighted_score[sp] = scores[sp] * len(self.filter_by_sp(self.test_set, sp).timeseries)
            scores_summary[sp] = scores[sp]

        average = np.sum(list(weighted_score.values())) / len(self.test_set.timeseries)
        scores_summary['Average'] = average

        return scores_summary

    @staticmethod
    def filter_by_sp(bundle: TimeseriesBundle, seasonal_pattern: str) -> TimeseriesBundle:
        return bundle.filter(lambda ts: ts.meta['seasonal_pattern'] == seasonal_pattern)

    @staticmethod
    def round_values(scores: OrderedDict):
        rounded_scores = OrderedDict()
        for k, v in scores.items():
            rounded_scores[k] = round_half_up(v, 3)
        return rounded_scores
