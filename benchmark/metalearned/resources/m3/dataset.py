import os
from dataclasses import dataclass
from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from common.settings import RESOURCES_DIR
from common.timeseries import Timeseries, TimeseriesBundle, TimeseriesLoader, Unknown, Year, Month
from common.utils import download_url


@dataclass(frozen=True)
class M3Meta:
    dataset_path = os.path.join(RESOURCES_DIR, 'm3')
    forecasts_path = os.path.join(dataset_path, 'forecasts')
    seasonal_patterns = ['M3Year', 'M3Quart', 'M3Month', 'M3Other']
    horizons = [6, 8, 18, 8]
    frequency = [1, 4, 12, 1]
    models = ['NAIVE2', 'SINGLE', 'HOLT', 'DAMPEN', 'WINTER', 'COMB S-H-D', 'B-J auto', 'AutoBox1', 'AutoBox2',
              'AutoBox3', 'ROBUST-Trend', 'ARARMA', 'Auto-ANN', 'Flors-Pearc1', 'Flors-Pearc2', 'PP-Autocast',
              'ForecastPro', 'SMARTFCS', 'THETAsm', 'THETA', 'RBF', 'ForcX']

    def horizons_map(self):
        return dict(zip(self.seasonal_patterns, self.horizons))

    def frequency_map(self):
        return dict(zip(self.seasonal_patterns, self.frequency))


class M3Dataset(TimeseriesLoader):
    def download(self) -> TimeseriesBundle:
        dataset_url = 'https://forecasters.org/data/m3comp/M3C.xls'
        raw_dataset_path = os.path.join(self.path, 'M3C.xls')

        download_url(dataset_url, raw_dataset_path)

        timeseries = []

        for sp in ['M3Year', 'M3Quart', 'M3Month', 'M3Other']:
            dataset = pd.read_excel(raw_dataset_path, sheet_name=sp)

            for _, row in dataset.iterrows():
                frequency = 1
                starting_date = Unknown.date()
                time_unit = Unknown()
                year = month = day = 1

                if 'Starting Year' in row.index:
                    year = row['Starting Year']
                    time_unit = Year()

                if 'Starting Quarter' in row.index:
                    month = 3 * (int(row['Starting Quarter']) - 1) + 1
                    frequency = 3
                    time_unit = Month()
                elif 'Starting Month' in row.index:
                    month = int(row['Starting Month'])
                    time_unit = Month()

                if not isinstance(time_unit, Unknown):
                    try:
                        starting_date = datetime(year=year, month=month, day=day)
                    except Exception:
                        time_unit = Unknown()
                        pass

                timeseries.append(Timeseries(id=str(row['Series']),
                                             start_date=starting_date,
                                             time_unit=time_unit,
                                             frequency=frequency,
                                             period=1,
                                             values=row.T[6:row.N + 6].values.astype(np.float32),
                                             meta={'seasonal_pattern': sp}
                                             ))
        return TimeseriesBundle(timeseries)

    def standard_split(self) -> Tuple[TimeseriesBundle, TimeseriesBundle]:
        bundle = self.load_cache()
        horizons_map = M3Meta().horizons_map()
        return bundle.split(lambda ts: ts.split(-horizons_map[ts.meta['seasonal_pattern']]))


class M3Forecasts(TimeseriesLoader):
    def download(self) -> TimeseriesBundle:
        raw_file_path = os.path.join(M3Meta.forecasts_path, 'M3Forecast.xls')
        download_url('https://forecasters.org/data/m3comp/M3Forecast.xls', raw_file_path)

        original_timeseries = M3Dataset(M3Meta().dataset_path).load_cache()
        horizon_mapping = M3Meta().horizons_map()
        training_set, _ = original_timeseries.split(lambda t: t.split(-horizon_mapping[t.meta['seasonal_pattern']]))
        training_timeseries = training_set.timeseries

        models_forecasts = []
        for model_name in tqdm(M3Meta.models):
            forecast = pd.read_excel(raw_file_path, sheet_name=model_name, header=None)
            for i, row in forecast.iterrows():
                ts = training_timeseries[i].future_values(row.T[2:row[1] + 2].values.astype(np.float32))
                ts.meta = {**ts.meta, 'model': model_name}
                models_forecasts.append(ts)
        return TimeseriesBundle(models_forecasts)


if __name__ == '__main__':
    M3Dataset(M3Meta.dataset_path).build_cache()
    M3Forecasts(M3Meta.forecasts_path).build_cache()
