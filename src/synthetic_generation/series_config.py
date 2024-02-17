"""
Module containing dataclasses for synthetic data generator
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class ComponentScale:
    base: float
    linear: float = None
    exp: float = None
    a: np.ndarray = None
    q: np.ndarray = None
    m: np.ndarray = None
    w: np.ndarray = None
    h: np.ndarray = None
    minute: np.ndarray = None


@dataclass
class ComponentNoise:
    # shape parameter for the weibull distribution
    k: float
    median: float

    # noise will be finally calculated as
    # noise_term = (1 + scale * (noise - E(noise)))
    # no noise can be represented by scale = 0
    scale: float


@dataclass
class SeriesConfig:
    scale: ComponentScale
    offset: ComponentScale
    noise_config: ComponentNoise

    def __str__(self):
        return f"L{1000*self.scale.linear:+02.0f}E{10000*(self.scale.exp - 1):+02.0f}A{100*self.scale.a:02.0f}M{100*self.scale.m:02.0f}W{100*self.scale.w:02.0f}"
