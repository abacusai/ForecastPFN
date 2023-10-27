import os
import ssl
from dataclasses import dataclass
from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd
import patoolib
from tqdm import tqdm

from common.settings import RESOURCES_DIR
from common.timeseries import Timeseries, TimeseriesBundle, TimeseriesLoader, Year, Month, Day, Hour
from common.utils import download_url


@dataclass(frozen=True)
class M4Meta:
    dataset_path = os.path.join(RESOURCES_DIR, 'm4')
    seasonal_patterns = ['Yearly', 'Quarterly', 'Monthly', 'Weekly', 'Daily', 'Hourly']
    horizons = [6, 8, 18, 13, 14, 48]
    period = [1, 4, 12, 1, 1, 24]

    def horizons_map(self):
        return dict(zip(self.seasonal_patterns, self.horizons))

    def period_map(self):
        return dict(zip(self.seasonal_patterns, self.period))


class M4Dataset(TimeseriesLoader):
    def download(self) -> TimeseriesBundle:
        url_template = 'https://github.com/Mcompetitions/M4-methods/raw/master/Dataset/{}/{}-{}.csv'
        m4_info_url = 'https://github.com/Mcompetitions/M4-methods/raw/master/Dataset/M4-info.csv'
        m4_info_path = os.path.join(self.path, 'M4info.csv')

        ssl._create_default_https_context = ssl._create_unverified_context

        download_url(m4_info_url, m4_info_path)
        for sp in M4Meta.seasonal_patterns:
            training_url = url_template.format("Train", sp, "train")
            download_url(training_url, os.path.join(M4Meta.dataset_path, f'{sp}-train.csv'))
            test_url = url_template.format("Test", sp, "test")
            download_url(test_url, os.path.join(M4Meta.dataset_path, f'{sp}-test.csv'))

        # Download naive2 forecasts, needed for OWA metric
        m4_naive2_archive = os.path.join(self.path, 'naive2.rar')
        download_url('https://github.com/M4Competition/M4-methods/raw/master/Point%20Forecasts/submission-Naive2.rar',
                     m4_naive2_archive)
        patoolib.extract_archive(m4_naive2_archive, outdir=self.path)
        os.remove(m4_naive2_archive)

        # Download m4 competition winner predictions, for summary testing purposes only
        m4_winner_archive = os.path.join(self.path, 'submission-118.rar')
        download_url('https://github.com/M4Competition/M4-methods/raw/master/Point%20Forecasts/submission-118.rar',
                     m4_winner_archive)
        patoolib.extract_archive(m4_winner_archive, outdir=self.path)
        os.remove(m4_winner_archive)

        m4_info = pd.read_csv(m4_info_path)
        m4_info.set_index('M4id', inplace=True)

        time_units_mapping = {
            'Yearly': (Year(), 1),
            'Quarterly': (Month(), 3),
            'Monthly': (Month(), 1),
            'Weekly': (Day(), 7),
            'Daily': (Day(), 1),
            'Hourly': (Hour(), 1)
        }

        all_timeseries = []
        for sp in M4Meta.seasonal_patterns:
            training_set = pd.read_csv(os.path.join(M4Meta.dataset_path, f'{sp}-train.csv'))
            test_set = pd.read_csv(os.path.join(M4Meta.dataset_path, f'{sp}-test.csv'))

            time_unit, frequency = time_units_mapping[sp]

            for i, row in tqdm(training_set.iterrows()):
                timeseries_id = str(row['V1'])
                training_values = row.values[1:].astype(np.float32)
                training_values = training_values[~np.isnan(training_values)]

                test_values = test_set.loc[i].values[1:].astype(np.float32)

                timeseries_info = m4_info.loc[timeseries_id]

                parsing_formats = ['%d-%m-%y %H:%M', '%Y-%m-%d %H:%M:%S']
                parsed_date = None
                for parsing_format in parsing_formats:
                    try:
                        parsed_date = datetime.strptime(timeseries_info.StartingDate, parsing_format)
                    except Exception:
                        continue
                if parsed_date is None:
                    raise ValueError(f'Could not parse {timeseries_info.StartingDate} for {timeseries_id}')
                # all M4 years are in the 1900s or 1800s
                if parsed_date.year > 2000:
                    parsed_date = parsed_date.replace(year=parsed_date.year - 100)

                timeseries = Timeseries(id=timeseries_id,
                                        start_date=parsed_date,
                                        time_unit=time_unit,
                                        frequency=frequency,
                                        period=int(timeseries_info.Frequency),
                                        values=np.concatenate([training_values, test_values]),
                                        meta={'seasonal_pattern': sp}
                                        )
                all_timeseries.append(timeseries)

        return TimeseriesBundle(all_timeseries)

    def standard_split(self) -> Tuple[TimeseriesBundle, TimeseriesBundle]:
        bundle = self.load_cache()
        horizons_map = M4Meta().horizons_map()
        return bundle.split(lambda ts: ts.split(-horizons_map[ts.meta['seasonal_pattern']]))

    @staticmethod
    def filter(bundle: TimeseriesBundle, seasonal_pattern: str) -> TimeseriesBundle:
        return bundle.filter(lambda ts: ts.meta['seasonal_pattern'] == seasonal_pattern)


if __name__ == '__main__':
    M4Dataset(M4Meta.dataset_path).build_cache()
