"""
Module to generate synthetic series
"""

from datetime import date

import numpy as np
import pandas as pd
from config_variables import Config
from constants import BASE_END, BASE_START, CONTEXT_LENGTH
from generate_series_components import make_series
from pandas.tseries.frequencies import to_offset
from scipy.stats import beta
from series_config import ComponentNoise, ComponentScale, SeriesConfig
from utils import get_transition_coefficients, sample_scale


def __generate(
    n=100,
    freq_index: int = None,
    start: pd.Timestamp = None,
    options: dict = {},
    random_walk: bool = False,
):
    """
    Function to construct synthetic series configs and generate
    synthetic series
    """
    if freq_index is None:
        freq_index = np.random.choice(len(Config.frequencies))

    freq, timescale = Config.frequencies[freq_index]

    # annual, monthly, weekly, hourly and minutely components
    a, m, w, h, minute = 0.0, 0.0, 0.0, 0.0, 0.0
    if freq == "min":
        minute = np.random.uniform(0.0, 1.0)
        h = np.random.uniform(0.0, 0.2)
    elif freq == "H":
        minute = np.random.uniform(0.0, 0.2)
        h = np.random.uniform(0.0, 1)
    elif freq == "D":
        w = np.random.uniform(0.0, 1.0)
        m = np.random.uniform(0.0, 0.2)
    elif freq == "W":
        m = np.random.uniform(0.0, 0.3)
        a = np.random.uniform(0.0, 0.3)
    elif freq == "MS":
        w = np.random.uniform(0.0, 0.1)
        a = np.random.uniform(0.0, 0.5)
    elif freq == "Y":
        w = np.random.uniform(0.0, 0.2)
        a = np.random.uniform(0.0, 1)
    else:
        raise NotImplementedError

    if start is None:
        # start = pd.Timestamp(date.fromordinal(np.random.randint(BASE_START, BASE_END)))
        start = pd.Timestamp(
            date.fromordinal(int((BASE_START - BASE_END) * beta.rvs(5, 1) + BASE_START))
        )

    scale_config = ComponentScale(
        1.0,
        np.random.normal(0, 0.01),
        np.random.normal(1, 0.005 / timescale),
        a=a,
        m=m,
        w=w,
        minute=minute,
        h=h,
    )

    offset_config = ComponentScale(
        0,
        np.random.uniform(-0.1, 0.5),
        np.random.uniform(-0.1, 0.5),
        a=np.random.uniform(0.0, 1.0),
        m=np.random.uniform(0.0, 1.0),
        w=np.random.uniform(0.0, 1.0),
    )

    noise_config = ComponentNoise(
        k=np.random.uniform(1, 5), median=1, scale=sample_scale()
    )

    cfg = SeriesConfig(scale_config, offset_config, noise_config)

    return cfg, make_series(cfg, to_offset(freq), n, start, options, random_walk)


def generate(
    n=100,
    freq_index: int = None,
    start: pd.Timestamp = None,
    options: dict = {},
    random_walk: bool = False,
):
    """
    Function to generate a synthetic series for a given config
    """

    cfg1, series1 = __generate(n, freq_index, start, options, random_walk)
    cfg2, series2 = __generate(n, freq_index, start, options, random_walk)

    if Config.transition:
        coeff = get_transition_coefficients(CONTEXT_LENGTH)
        values = coeff * series1["values"] + (1 - coeff) * series2["values"]
    else:
        values = series1["values"]

    dataframe_data = {"series_values": values, "noise": series1["noise"]}

    return cfg1, pd.DataFrame(
        data=dataframe_data, index=series1["dates"]
    )  # .clip(lower=0.0)
