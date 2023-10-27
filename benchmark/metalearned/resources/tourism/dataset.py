import os
from dataclasses import dataclass
from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd
import patoolib

from common.settings import RESOURCES_DIR
from common.timeseries import Timeseries, TimeseriesBundle, TimeseriesLoader, Year, Month
from common.utils import download_url


@dataclass(frozen=True)
class TourismMeta:
    dataset_path = os.path.join(RESOURCES_DIR, 'tourism')
    seasonal_patterns = ['Yearly', 'Quarterly', 'Monthly']
    horizons = [4, 8, 24]
    period = [1, 4, 12]

    def horizons_map(self):
        return dict(zip(self.seasonal_patterns, self.horizons))

    def period_map(self):
        return dict(zip(self.seasonal_patterns, self.period))


class TourismDataset(TimeseriesLoader):
    def download(self) -> TimeseriesBundle:
        archive_file = os.path.join(self.path, 'm3.zip')
        download_url('https://robjhyndman.com/data/27-3-Athanasopoulos1.zip', archive_file)
        patoolib.extract_archive(archive_file, outdir=self.path)

        timeseries = []

        # Yearly
        insample = pd.read_csv(os.path.join(TourismMeta.dataset_path, f'yearly_in.csv'), header=0)
        outsample = pd.read_csv(os.path.join(TourismMeta.dataset_path, f'yearly_oos.csv'), header=0)
        outsampleT = outsample.T

        for timeseries_id, ts_row in insample.T.iterrows():
            outsample_row = outsampleT.loc[timeseries_id].values
            start_date = datetime.strptime(str(int(ts_row[[1]])), '%Y')
            insample_values = ts_row.values[2:2 + int(ts_row[[0]])]
            outsample_values = outsample_row[2:2 + int(outsample_row[0])]
            values = np.concatenate([insample_values, outsample_values])
            timeseries.append(Timeseries(id=timeseries_id,
                                         start_date=start_date,
                                         time_unit=Year(),
                                         frequency=1,
                                         period=1,
                                         values=values,
                                         meta={'seasonal_pattern': 'Yearly'}))

        # Quarterly
        insample = pd.read_csv(os.path.join(TourismMeta.dataset_path, f'quarterly_in.csv'), header=0)
        outsample = pd.read_csv(os.path.join(TourismMeta.dataset_path, f'quarterly_oos.csv'), header=0)
        outsampleT = outsample.T

        for timeseries_id, ts_row in insample.T.iterrows():
            outsample_row = outsampleT.loc[timeseries_id].values
            start_date = datetime.strptime(f'{str(int(ts_row[[1]]))}-{str((int(ts_row[[2]]) - 1) * 3)}', '%Y-%M')
            insample_values = ts_row.values[3:3 + int(ts_row[[0]])]
            outsample_values = outsample_row[3:3 + int(outsample_row[0])]
            values = np.concatenate([insample_values, outsample_values])
            timeseries.append(Timeseries(id=timeseries_id,
                                         start_date=start_date,
                                         time_unit=Month(),
                                         frequency=3,
                                         period=1,
                                         values=values,
                                         meta={'seasonal_pattern': 'Quarterly'}))

        # Monthly
        insample = pd.read_csv(os.path.join(TourismMeta.dataset_path, f'monthly_in.csv'), header=0)
        outsample = pd.read_csv(os.path.join(TourismMeta.dataset_path, f'monthly_oos.csv'), header=0)
        outsampleT = outsample.T

        for timeseries_id, ts_row in insample.T.iterrows():
            outsample_row = outsampleT.loc[timeseries_id].values
            start_date = datetime.strptime(f'{str(int(ts_row[[1]]))}-{str(int(ts_row[[2]]))}', '%Y-%M')
            insample_values = ts_row.values[3:3 + int(ts_row[[0]])]
            outsample_values = outsample_row[3:3 + int(outsample_row[0])]
            values = np.concatenate([insample_values, outsample_values])
            timeseries.append(Timeseries(id=timeseries_id,
                                         start_date=start_date,
                                         time_unit=Month(),
                                         frequency=1,
                                         period=1,
                                         values=values,
                                         meta={'seasonal_pattern': 'Monthly'}))

        return TimeseriesBundle(timeseries)

    def standard_split(self) -> Tuple[TimeseriesBundle, TimeseriesBundle]:
        bundle = self.load_cache()
        horizons_map = TourismMeta().horizons_map()
        return bundle.split(lambda ts: ts.split(-horizons_map[ts.meta['seasonal_pattern']]))


if __name__ == '__main__':
    TourismDataset(TourismMeta.dataset_path).build_cache()
