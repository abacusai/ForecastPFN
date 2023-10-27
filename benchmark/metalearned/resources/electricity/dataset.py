import os
from dataclasses import dataclass
from datetime import datetime
from typing import Tuple

import numpy as np
import patoolib
from tqdm import tqdm

from common.settings import RESOURCES_DIR
from common.timeseries import Timeseries, TimeseriesBundle, TimeseriesLoader, Hour
from common.utils import download_url

"""
Hourly aggregated dataset from https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014

As it is used in https://www.cs.utexas.edu/~rofuyu/papers/tr-mf-nips.pdf
Dataset was also compared with the one built by the TRMF paper's author:
https://github.com/rofuyu/exp-trmf-nips16/blob/master/python/exp-scripts/datasets/download-data.sh
"""


@dataclass(frozen=True)
class ElectricityMeta:
    dataset_path = os.path.join(RESOURCES_DIR, 'electricity')
    horizon = 24
    windows = 7
    clients = 370
    time_steps = 26304
    seasonal_pattern = 'Hourly'
    period = 24
    # first point of test set as per https://arxiv.org/pdf/1704.04110.pdf
    deepar_split = datetime(2014, 9, 1, 0)
    # first point of test set as per Figure B.5 http://proceedings.mlr.press/v97/wang19k/wang19k-supp.pdf
    deepfact_split = datetime(2014, 3, 31, 1)


class ElectricityDataset(TimeseriesLoader):
    def download(self) -> TimeseriesBundle:
        archive_file = os.path.join(self.path, 'dataset.zip')
        raw_file = os.path.join(self.path, 'LD2011_2014.txt')
        download_url('https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip',
                     archive_file)
        patoolib.extract_archive(archive_file, outdir=self.path)

        with open(raw_file, 'r') as f:
            raw = f.readlines()

        parsed_values = np.array(list(map(
            lambda raw_line: np.array(raw_line.replace(',', '.').strip().split(';')[1:]).astype(np.float), tqdm(raw[1:])
        )))

        aggregated = []
        for i in tqdm(range(0, parsed_values.shape[0], 4)):
            aggregated.append(parsed_values[i:i + 4, :].sum(axis=0))
        aggregated = np.array(aggregated)

        # regarding time labels, in dataset description authors specify
        # "Every year in March time change day (which has only 23 hours) the values between 1:00 am and 2:00 am
        # are zero for all points."
        # But I could not prove that claim for "2011-03-27 01:15:00" (lines 8165-8167),
        # neither for "2012-03-25 01:45:00", thus it's not clear how to deal with daylight saving time change in this
        # dataset. Taking into account this uncertainty the starting date is treated as UTC (without time changes).

        start_date = datetime(2011, 1, 1, 1, 0, 0)  # aggregated towards next hour instead of current hour.

        dataset = aggregated.T  # use time step as second dimension.
        timeseries = []

        for i, values in enumerate(dataset):
            timeseries.append(Timeseries(id=str(i),
                                         start_date=start_date,
                                         time_unit=Hour(),
                                         frequency=1,
                                         period=ElectricityMeta.period,
                                         values=values,
                                         meta={}))
        return TimeseriesBundle(timeseries)

    def standard_split(self) -> Tuple[TimeseriesBundle, TimeseriesBundle]:
        bundle = self.load_cache()
        return bundle.split(lambda ts: ts.split(-24 * 7))


if __name__ == '__main__':
    ElectricityDataset(ElectricityMeta.dataset_path).build_cache()
