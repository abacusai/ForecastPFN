import os
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
import pandas as pd

from common.evaluator import Evaluator, EvaluationResult
from common.metrics import mase, smape_2
from common.timeseries import TimeseriesBundle
from common.utils import clean_nans, round_half_up
from resources.m4.dataset import M4Dataset, M4Meta


@dataclass
class M4Evaluator(Evaluator):
    validation: bool = False
    metric: str = 'sMAPE'

    def evaluate(self, forecast: TimeseriesBundle) -> EvaluationResult:
        insamples, _ = M4Dataset(M4Meta.dataset_path).standard_split()
        if self.validation:
            horizons_map = M4Meta().horizons_map()
            insamples, _ = insamples.split(lambda ts: ts.split(-horizons_map[ts.meta['seasonal_pattern']]))

        grouped_smapes = {sp: np.mean(smape_2(forecast=np.array(M4Dataset.filter(forecast, sp).values()),
                                              target=np.array(M4Dataset.filter(self.test_set, sp).values())))
                          for sp in M4Meta.seasonal_patterns}

        grouped_smapes = self.summarize_groups(grouped_smapes)

        if self.metric == 'OWA':
            grouped_owa = OrderedDict()
            if not self.validation:
                naive2_forecasts = pd.read_csv(
                    os.path.join(M4Meta.dataset_path, 'submission-Naive2.csv'))
                naive2_forecasts.set_index(keys='id', inplace=True)

                model_mases = {}
                naive2_smapes = {}
                naive2_mases = {}
                for sp in M4Meta.seasonal_patterns:
                    model_forecasts = M4Dataset.filter(forecast, sp)
                    naive2_forecast = clean_nans(naive2_forecasts.loc[model_forecasts.ids()].values)

                    model_forecast_values = model_forecasts.values()

                    target = self.test_set.filter(lambda ts: ts.meta['seasonal_pattern'] == sp)
                    target_values = np.array(target.values())
                    # all timeseries within group have same frequency
                    period = target.period()[0]
                    insample = M4Dataset.filter(insamples, sp).values()

                    model_mases[sp] = np.mean([mase(forecast=model_forecast_values[i],
                                                    insample=insample[i],
                                                    outsample=target_values[i],
                                                    frequency=period) for i in range(len(model_forecast_values))])
                    naive2_mases[sp] = np.mean([mase(forecast=naive2_forecast[i],
                                                     insample=insample[i],
                                                     outsample=target_values[i],
                                                     frequency=period) for i in range(len(model_forecast_values))])

                    naive2_smapes[sp] = np.mean(smape_2(naive2_forecast, target_values))
                grouped_model_mases = self.summarize_groups(model_mases)
                grouped_naive2_smapes = self.summarize_groups(naive2_smapes)
                grouped_naive2_mases = self.summarize_groups(naive2_mases)
                for k in grouped_model_mases.keys():
                    grouped_owa[k] = round_half_up((grouped_model_mases[k] / grouped_naive2_mases[k] +
                                                    grouped_smapes[k] / grouped_naive2_smapes[k]) / 2, 3)
            return self.round_values(grouped_owa)
        else:
            return self.round_values(grouped_smapes)

    def summarize_groups(self, scores):
        scores_summary = OrderedDict()

        weighted_score = {}
        for sp in ['Yearly', 'Quarterly', 'Monthly', 'Weekly', 'Daily', 'Hourly']:
            weighted_score[sp] = scores[sp] * len(M4Dataset.filter(self.test_set, sp).timeseries)
            scores_summary[sp] = scores[sp]

        others_score = 0
        others_count = 0
        for sp in ['Weekly', 'Daily', 'Hourly']:
            number_of_timeseries = len(M4Dataset.filter(self.test_set, sp).timeseries)
            others_score += scores[sp] * number_of_timeseries
            others_count += number_of_timeseries
        weighted_score['Others'] = others_score
        scores_summary['Others'] = others_score / others_count

        others_score = 0
        others_count = 0
        for sp in ['Daily', 'Weekly']:
            number_of_timeseries = len(M4Dataset.filter(self.test_set, sp).timeseries)
            others_score += scores[sp] * number_of_timeseries
            others_count += number_of_timeseries
        weighted_score['D+W'] = others_score
        scores_summary['D+W'] = others_score / others_count

        others_score = 0
        others_count = 0
        for sp in ['Daily', 'Weekly', 'Monthly']:
            number_of_timeseries = len(M4Dataset.filter(self.test_set, sp).timeseries)
            others_score += scores[sp] * number_of_timeseries
            others_count += number_of_timeseries
        weighted_score['D+W+M'] = others_score
        scores_summary['D+W+M'] = others_score / others_count
        others_score = 0
        others_count = 0
        for sp in ['Daily', 'Weekly', 'Monthly', 'Yearly']:
            number_of_timeseries = len(M4Dataset.filter(self.test_set, sp).timeseries)
            others_score += scores[sp] * number_of_timeseries
            others_count += number_of_timeseries
        weighted_score['D+W+M+Y'] = others_score
        scores_summary['D+W+M+Y'] = others_score / others_count

        average = np.sum(list(weighted_score.values())) / len(self.test_set.timeseries)
        scores_summary['Average'] = average

        return scores_summary

    @staticmethod
    def round_values(scores: OrderedDict):
        rounded_scores = OrderedDict()
        for k, v in scores.items():
            rounded_scores[k] = round_half_up(v, 3)
        return rounded_scores
