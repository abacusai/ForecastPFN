import logging
import os
import pickle
from dataclasses import dataclass
from typing import Tuple

from tqdm import tqdm

from common.settings import RESOURCES_DIR
from common.timeseries import Timeseries, TimeseriesBundle, TimeseriesLoader, Year, Month, Day
from resources.fred.api import FredAPI


@dataclass(frozen=True)
class FredMeta:
    dataset_path = os.path.join(RESOURCES_DIR, 'fred')
    seasonal_patterns = ['Yearly', 'Quarterly', 'Monthly', 'Weekly', 'Daily']
    horizons = [6, 8, 18, 13, 14]
    period = [1, 4, 12, 1, 1]

    def horizons_map(self):
        return dict(zip(self.seasonal_patterns, self.horizons))

    def period_map(self):
        return dict(zip(self.seasonal_patterns, self.period))


class FredDataset(TimeseriesLoader):
    def download(self) -> TimeseriesBundle:
        FredAPI(self.path).fetch_all()
        with open(os.path.join(self.path, 'dataset.pickle'), 'rb') as f:
            raw_data = pickle.load(f)['data']

        frequency_map = {
            'Yearly': (Year(), 1),
            'Quarterly': (Month(), 3),
            'Monthly': (Month(), 1),
            'Weekly': (Day(), 7),
            'Daily': (Day(), 1)
        }

        period_map = FredMeta().period_map()

        timeseries = []
        for ts_id, record in tqdm(raw_data.items()):
            sp = record['time_unit']
            frequency = [frequency_map[s] for s in frequency_map.keys() if sp.startswith(s)]
            period = [period_map[s] for s in period_map.keys() if sp.startswith(s)]
            if len(frequency) > 0:
                frequency = frequency[0]
            else:
                raise Exception(f"Cannot match frequency for: {sp}")
            if len(period) > 0:
                period = period[0]
            else:
                raise Exception(f"Cannot match frequency for: {sp}")
            timeseries.append(Timeseries(id=ts_id,
                                         start_date=record['start_date'],
                                         time_unit=frequency[0],
                                         frequency=frequency[1],
                                         period=period,
                                         values=record['values'],
                                         meta={'seasonal_pattern': sp}
                                         ))
        grouped_timeseries = [list(filter(lambda ts: ts.meta['seasonal_pattern'] == sp, timeseries))
                              for sp in FredMeta.seasonal_patterns]
        grouped_timeseries = [ts for sp_ts in grouped_timeseries for ts in sp_ts]

        return TimeseriesBundle(grouped_timeseries)

    def standard_split(self) -> Tuple[TimeseriesBundle, TimeseriesBundle]:
        bundle = self.load_cache()
        horizons_map = FredMeta().horizons_map()
        return bundle.split(lambda ts: ts.split(-horizons_map[ts.meta['seasonal_pattern']]))


if __name__ == '__main__':
    logging.root.setLevel(logging.INFO)
    FredDataset(FredMeta.dataset_path).build_cache()
