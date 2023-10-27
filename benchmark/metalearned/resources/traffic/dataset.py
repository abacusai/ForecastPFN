import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Tuple

import numpy as np
import patoolib
from tqdm import tqdm

from common.settings import RESOURCES_DIR
from common.timeseries import Timeseries, TimeseriesBundle, TimeseriesLoader, Hour
from common.utils import download_url


"""
Hourly aggregated dataset from https://archive.ics.uci.edu/ml/datasets/PEMS-SF

As it is used in https://www.cs.utexas.edu/~rofuyu/papers/tr-mf-nips.pdf
Dataset was also compared with the one built by the TRMF paper's author:
https://github.com/rofuyu/exp-trmf-nips16/blob/master/python/exp-scripts/datasets/download-data.sh
"""
@dataclass(frozen=True)
class TrafficMeta:
    dataset_path = os.path.join(RESOURCES_DIR, 'traffic')
    horizon = 24
    stations = 963
    seasonal_pattern = 'Hourly'
    period = 24 * 7
    # as per https://arxiv.org/pdf/1704.04110.pdf
    deepar_split = datetime(2008, 6, 15, 0)
    # as per Figure B.5 http://proceedings.mlr.press/v97/wang19k/wang19k-supp.pdf
    deepfact_split = datetime(2008, 1, 13, 18)


class TrafficDataset(TimeseriesLoader):
    def download(self) -> TimeseriesBundle:
        archive_file = os.path.join(self.path, 'dataset.zip')
        train_raw_file = os.path.join(self.path, 'PEMS_train')
        test_raw_file = os.path.join(self.path, 'PEMS_test')
        perm_raw_file = os.path.join(self.path, 'randperm')
        download_url('https://archive.ics.uci.edu/ml/machine-learning-databases/00204/PEMS-SF.zip',
                     archive_file)
        patoolib.extract_archive(archive_file, outdir=self.path)
        with open(train_raw_file, 'r') as f:
            train_raw_data = f.readlines()
        with open(test_raw_file, 'r') as f:
            test_raw_data = f.readlines()
        with open(perm_raw_file, 'r') as f:
            permutations = f.readlines()
        permutations = np.array(permutations[0].rstrip()[1:-1].split(' ')).astype(np.int)

        raw_data = train_raw_data + test_raw_data

        # start date per https://archive.ics.uci.edu/ml/datasets/PEMS-SF
        # skip 2008-01-01 because it's holiday.
        # the number of days between 2008-01-01 and 2009-03-30 is 455 but based on provided labels (which are days of week)
        # the sequence of days had only 10 gaps by 1 day, where the first 6 correspond to a holiday or anomalous day which
        # was excluded from the dataset, but the other 4 gaps happen on unexplained dates.
        # More over with only 10 gaps it's not possible to fill dates up to 2009-03-30, it should be 15 gaps
        # (if 2009-01-01 included, 14 otherwise).
        # Taking into consideration all the concerns above, we decided to assume the following dates were skipped
        # (first 7 seem to be aligned with labels and description):
        #  - Jan. 1, 2008
        #  - Jan. 21, 2008
        #  - Feb. 18, 2008
        #  - Mar. 9, 2008 - Anomaly
        #  - May 26, 2008
        #  - Jul. 4, 2008
        #  - Sep. 1, 2008
        #  - Oct. 13, 2008 - Columbus Day
        #  - Nov. 11, 2008
        #  - Nov. 27, 2008
        #  - Dec. 25, 2008
        #  - Jan. 1, 2009
        #  - Jan. 19, 2009
        #  - Feb. 16, 2009
        #  - Mar. 8, 2009 - Anomaly
        #  ------------------------------------------
        # Thus 455 - 15 = 440 days from 2008-01-01 to 2008-03-30 (incl.)
        start_date = datetime.strptime('2008-01-02', '%Y-%m-%d')  # 2008-01-01 is a holiday
        current_date = start_date
        excluded_dates = [
            datetime.strptime('2008-01-21', '%Y-%m-%d'),
            datetime.strptime('2008-02-18', '%Y-%m-%d'),
            datetime.strptime('2008-03-09', '%Y-%m-%d'),
            datetime.strptime('2008-05-26', '%Y-%m-%d'),
            datetime.strptime('2008-07-04', '%Y-%m-%d'),
            datetime.strptime('2008-09-01', '%Y-%m-%d'),
            datetime.strptime('2008-10-13', '%Y-%m-%d'),
            datetime.strptime('2008-11-11', '%Y-%m-%d'),
            datetime.strptime('2008-11-27', '%Y-%m-%d'),
            datetime.strptime('2008-12-25', '%Y-%m-%d'),
            datetime.strptime('2009-01-01', '%Y-%m-%d'),
            datetime.strptime('2009-01-19', '%Y-%m-%d'),
            datetime.strptime('2009-02-16', '%Y-%m-%d'),
            datetime.strptime('2009-03-08', '%Y-%m-%d'),
        ]

        values = []
        for day, i in tqdm(enumerate(range(len(permutations)))):
            if current_date not in excluded_dates:
                matrix = raw_data[np.where(permutations == i + 1)[0][0]].rstrip()[1:-1]
                daily = []
                for row_vector in matrix.split(';'):
                    daily.append(np.array(row_vector.split(' ')).astype(np.float32))
                daily = np.array(daily)
                if len(values) == 0:
                    values = daily
                else:
                    values = np.concatenate([values, daily], axis=1)
            else:  # should never be in the first 24*7 records.
                # fill gaps with same day of previous week.
                values = np.concatenate([values, values[:, -24 * 7 * 6:-24 * 6 * 6]], axis=1)
            current_date += timedelta(days=1)

        # aggregate 10 minutes events to hourly
        hourly = np.array([list(map(np.mean, zip(*(iter(lane),) * 6))) for lane in tqdm(values)])
        timeseries = [Timeseries(id=str(i),
                                 start_date=start_date,
                                 time_unit=Hour(),
                                 frequency=1,
                                 period=24 * 7,
                                 values=values,
                                 meta={}) for i, values in enumerate(hourly)]
        return TimeseriesBundle(timeseries=timeseries)

    def standard_split(self) -> Tuple[TimeseriesBundle, TimeseriesBundle]:
        bundle = self.load_cache()
        return bundle.split(lambda ts: ts.split(-24 * 7))


if __name__ == '__main__':
    TrafficDataset(TrafficMeta.dataset_path).build_cache()
