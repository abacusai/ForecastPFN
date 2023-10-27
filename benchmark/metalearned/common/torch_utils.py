import os
import tempfile
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch as t


def to_tensor(array: np.ndarray, use_cuda: bool = True):
    tensor = t.tensor(array, dtype=t.float32)
    if use_cuda and t.cuda.is_available():
        tensor = tensor.cuda()
    return tensor


def to_device(module: t.nn.Module, use_cuda: bool = True):
    return module.cuda() if use_cuda and t.cuda.is_available() else module


def div_no_nan(a, b):
    result = a / b
    result[result != result] = .0
    result[result == np.inf] = .0
    return result


def mape_loss(forecast, target, mask):
    weights = div_no_nan(mask, target)
    return t.mean(t.abs((forecast - target) * weights))


def smape_1_loss(forecast, target, mask):
    return 200 * t.mean(div_no_nan(t.abs(forecast - target), forecast.data + target.data) * mask)


def smape_2_loss(forecast, target, mask):
    return 200 * t.mean(div_no_nan(t.abs(forecast - target), t.abs(forecast.data) + t.abs(target.data)) * mask)


def mase_loss(insample, freq, forecast, target, mask):
    masep = t.mean(t.abs(insample[:, freq:] - insample[:, :-freq]), dim=1)
    masked_masep_inv = div_no_nan(mask, masep[:, None])
    return t.mean(t.abs(target - forecast) * masked_masep_inv)


class SnapshotManager:
    def __init__(self, snapshot_dir: str, logging_frequency: int, snapshot_frequency: int):
        self.model_snapshot_file = os.path.join(snapshot_dir, 'model')
        self.optimizer_snapshot_file = os.path.join(snapshot_dir, 'optimizer')
        self.losses_file = os.path.join(snapshot_dir, 'losses')
        self.iteration_file = os.path.join(snapshot_dir, 'iteration')
        self.time_tracking_file = os.path.join(snapshot_dir, 'time')
        self.logging_frequency = max(logging_frequency, 1)
        self.snapshot_frequency = max(snapshot_frequency, 1)
        self.start_time = None
        self.losses = {'training': {}, 'validation': {}}
        self.time_track = {}

    def restore(self, model: Optional[t.nn.Module], optimizer: Optional[t.optim.Optimizer]) -> int:
        if model is not None and os.path.isfile(self.model_snapshot_file):
            model.load_state_dict(t.load(self.model_snapshot_file))
        if optimizer is not None and os.path.isfile(self.optimizer_snapshot_file):
            optimizer.load_state_dict(t.load(self.optimizer_snapshot_file))
        iteration = t.load(self.iteration_file)['iteration'] if os.path.isfile(self.iteration_file) else 0
        if os.path.isfile(self.losses_file):
            losses = t.load(self.losses_file)
            training_losses = {k: v for k, v in losses['training'].items() if k <= iteration}
            validation_losses = {k: v for k, v in losses['validation'].items() if k <= iteration}
            # when restoring remove losses which were after the last snapshot
            self.losses = {'training': training_losses, 'validation': validation_losses}
            self.snapshot(self.losses_file, self.losses)
        if os.path.isfile(self.time_tracking_file):
            self.time_track = t.load(self.time_tracking_file)
        return iteration

    def load_training_losses(self) -> pd.DataFrame:
        if os.path.isfile(self.losses_file):
            losses = t.load(self.losses_file)['training']
            return pd.DataFrame(losses, index=[0])[sorted(losses.keys())].T
        else:
            return pd.DataFrame([np.nan])

    def enable_time_tracking(self):
        self.start_time = time.time()

    def register(self,
                 iteration: int,
                 training_loss: float,
                 validation_loss: float,
                 model: t.nn.Module,
                 optimizer: Optional[t.optim.Optimizer]) -> None:
        if iteration == 1 or iteration % self.logging_frequency == 0:
            self.losses['training'][iteration] = training_loss
            self.losses['validation'][iteration] = validation_loss
            self.snapshot(self.losses_file, self.losses)
        if iteration % self.snapshot_frequency == 0:
            self.snapshot(self.model_snapshot_file, model.state_dict())
            if optimizer is not None:
                self.snapshot(self.optimizer_snapshot_file, optimizer.state_dict())
            self.snapshot(self.iteration_file, {'iteration': iteration})
            if self.start_time is not None:
                self.time_track[iteration] = time.time() - self.start_time
                self.snapshot(self.time_tracking_file, self.time_track)
                self.start_time = time.time()

    @staticmethod
    def snapshot(path: str, data: Dict):
        dir_path = os.path.dirname(path)
        if not os.path.isdir(dir_path):
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        temp_file = tempfile.NamedTemporaryFile(dir=dir_path, delete=False, mode='wb')
        t.save(data, temp_file)
        temp_file.flush()
        os.fsync(temp_file.fileno())
        os.rename(temp_file.name, path)
