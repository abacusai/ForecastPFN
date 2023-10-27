import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
from dateutil.relativedelta import relativedelta
import dill


class TimeUnit(ABC):
    @abstractmethod
    def add(self, start_date: datetime, n: int) -> datetime:
        pass

    @abstractmethod
    def delta(self, date_1: datetime, date_2: datetime) -> float:
        pass


class Year(TimeUnit):
    def add(self, start_date: datetime, n: int) -> datetime:
        return start_date + relativedelta(years=n)

    def delta(self, date_1: datetime, date_2: datetime) -> float:
        return relativedelta(date_1, date_2).years


class Month(TimeUnit):
    def add(self, start_date: datetime, n: int) -> datetime:
        return start_date + relativedelta(months=n)

    def delta(self, date_1: datetime, date_2: datetime) -> float:
        relative_date = relativedelta(date_1, date_2)
        return relative_date.years * 12 + relative_date.months


class Day(TimeUnit):
    def add(self, start_date: datetime, n: int) -> datetime:
        return start_date + relativedelta(days=n)

    def delta(self, date_1: datetime, date_2: datetime) -> float:
        return (date_1 - date_2) / timedelta(days=1)


class Hour(TimeUnit):
    def add(self, start_date: datetime, n: int) -> datetime:
        return start_date + relativedelta(hours=n)

    def delta(self, date_1: datetime, date_2: datetime) -> float:
        return (date_1 - date_2) / timedelta(hours=1)


class Minute(TimeUnit):
    def add(self, start_date: datetime, n: int) -> datetime:
        return start_date + relativedelta(minutes=n)

    def delta(self, date_1: datetime, date_2: datetime) -> float:
        return (date_1 - date_2) / timedelta(minutes=1)


class Second(TimeUnit):
    def add(self, start_date: datetime, n: int) -> datetime:
        return start_date + relativedelta(seconds=n)

    def delta(self, date_1: datetime, date_2: datetime) -> float:
        return (date_1 - date_2) / timedelta(seconds=1)


class Unknown(TimeUnit):
    def add(self, start_date: datetime, n: int) -> datetime:
        return Unknown.date()

    def delta(self, date_1: datetime, date_2: datetime) -> float:
        return np.nan

    @staticmethod
    def date() -> datetime:
        return datetime(1, 1, 1)


TimeseriesSplit = Tuple['Timeseries', 'Timeseries']


@dataclass
class Timeseries:
    id: str
    # Datetime object, see: https://docs.python.org/3/library/datetime.html (naive or aware type depends on use cases).
    start_date: datetime
    time_unit: TimeUnit
    # Frequency meaning the rate at which the events are registered, in terms of frequency type (year, month, etc.).
    frequency: int
    # Period in terms of how many time units in a "cycle".
    period: int
    values: np.ndarray
    meta: Dict[str, Any]

    def copy(self, start_date: datetime, values: np.ndarray) -> 'Timeseries':
        return Timeseries(id=self.id,
                          start_date=start_date,
                          time_unit=self.time_unit,
                          frequency=self.frequency,
                          period=self.period,
                          values=values,
                          meta=self.meta)

    def future_values(self, values: np.ndarray) -> 'Timeseries':
        return self.copy(start_date=self.time_unit.add(self.start_date, len(self.values)), values=values)

    def split(self, n: int) -> TimeseriesSplit:
        time_shift = n if n >= 0 else len(self.values) + n
        split_time = self.time_unit.add(self.start_date, time_shift * self.frequency)
        return self.copy(start_date=self.start_date, values=self.values[:n]), self.copy(start_date=split_time,
                                                                                        values=self.values[n:])

    def split_by_time(self, split_date: datetime) -> TimeseriesSplit:
        points_to_include = int(self.time_unit.delta(split_date, self.start_date) // self.frequency)
        if points_to_include < 0:
            before = self.copy(split_date, np.empty(0))
            on_and_after = self
        else:
            before = self.copy(self.start_date, self.values[:points_to_include])
            on_and_after = self.copy(split_date, self.values[points_to_include:])
        return before, on_and_after


@dataclass
class TimeseriesBundle:
    timeseries: List[Timeseries]

    def values(self) -> List[np.ndarray]:
        return list(map(lambda ts: ts.values, self.timeseries))

    def time_stamps(self) -> List[np.ndarray]:
        def _make_time_stamps(ts):
            return np.array([ts.time_unit.add(ts.start_date, ts.frequency*i)
                    for i in range(len(ts.values))])

        return list(map(_make_time_stamps, self.timeseries))

    def period(self) -> List[int]:
        return list(map(lambda ts: ts.period, self.timeseries))

    def ids(self) -> List[str]:
        return list(map(lambda ts: ts.id, self.timeseries))

    def filter(self, f: Callable[[Timeseries], bool]) -> 'TimeseriesBundle':
        return TimeseriesBundle(list(filter(f, self.timeseries)))

    def map(self, f: Callable[[Timeseries], Timeseries]) -> 'TimeseriesBundle':
        return TimeseriesBundle(list(map(f, self.timeseries)))

    def split(self, f: Callable[[Timeseries], TimeseriesSplit]) -> Tuple['TimeseriesBundle', 'TimeseriesBundle']:
        bucket_1 = []
        bucket_2 = []
        for timeseries in self.timeseries:
            part_1, part_2 = f(timeseries)
            bucket_1.append(part_1)
            bucket_2.append(part_2)
        return TimeseriesBundle(bucket_1), TimeseriesBundle(bucket_2)

    def intersection_by_id(self, bundle: 'TimeseriesBundle') -> Tuple['TimeseriesBundle', 'TimeseriesBundle']:
        bundle_ids = bundle.ids()
        ids = [ts_id for ts_id in self.ids() if ts_id in bundle_ids]
        return self.filter(lambda ts: ts.id in ids), bundle.filter(lambda ts: ts.id in ids)

    def future_values(self, values: np.array) -> 'TimeseriesBundle':
        assert len(values) == len(self.timeseries)
        return TimeseriesBundle([ts.future_values(values[i]) for i, ts in enumerate(self.timeseries)])


class TimeseriesLoader(ABC):
    def __init__(self, path: str):
        self.path = path
        self.cache_path = os.path.join(path, 'cache.dill')

    def build_cache(self) -> None:
        if not os.path.exists(self.cache_path):
            Path(self.cache_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, 'wb') as file:
                dill.dump(self.download(), file)

    def load_cache(self) -> TimeseriesBundle:
        with open(self.cache_path, 'rb') as file:
            return dill.load(file)

    @abstractmethod
    def download(self) -> TimeseriesBundle:
        """
        :return: Training and test splits.
        """
        pass

