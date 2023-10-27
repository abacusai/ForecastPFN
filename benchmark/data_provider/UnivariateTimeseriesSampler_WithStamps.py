import numpy as np
import pandas as pd
import datetime

class UnivariateTimeseriesSampler_WithStamps:
    def __init__(self,
                 timeseries: np.ndarray,
                 time_stamps: np.ndarray,
                 insample_size: int,
                 outsample_size: int,
                 window_sampling_limit: int,
                 batch_size: int,
                 time_features,
                 ):
        self.timeseries = [ts for ts in timeseries]
        self.time_stamps = [ts for ts in time_stamps]
        self.window_sampling_limit = window_sampling_limit
        self.batch_size = batch_size
        self.insample_size = insample_size
        self.outsample_size = outsample_size
        self.time_features = time_features
        self.time_embedding_dim = self.time_features(self.time_stamps[0]).T.shape[0]
        

    def __iter__(self):
        while True:
            insample = np.zeros((self.batch_size, self.insample_size))
            insample_mask = np.zeros((self.batch_size, self.insample_size))
            outsample = np.zeros((self.batch_size, self.outsample_size))
            outsample_mask = np.zeros((self.batch_size, self.outsample_size))
            sampled_ts_indices = np.random.randint(len(self.timeseries), size=self.batch_size)

            insample_time_stamps = np.zeros(
                (self.batch_size, self.insample_size, self.time_embedding_dim), dtype=object)
            outsample_time_stamps = np.zeros(
                (self.batch_size, self.outsample_size, self.time_embedding_dim), dtype=object)
            for i, sampled_index in enumerate(sampled_ts_indices):
                sampled_timeseries = self.timeseries[sampled_index]
                cut_point = np.random.randint(low=max(1, len(sampled_timeseries) - self.window_sampling_limit),
                                              high=len(sampled_timeseries),
                                              size=1)[0]

                insample_window = sampled_timeseries[max(0, cut_point - self.insample_size):cut_point]
                insample[i, -len(insample_window):] = insample_window
                insample_mask[i, -len(insample_window):] = 1.0
                outsample_window = sampled_timeseries[
                                   cut_point:min(len(sampled_timeseries), cut_point + self.outsample_size)]
                outsample[i, :len(outsample_window)] = outsample_window
                outsample_mask[i, :len(outsample_window)] = 1.0

                sampled_timestamps = self.time_stamps[sampled_index]
                insample_window_time_stamps = sampled_timestamps[max(0, cut_point - self.insample_size):cut_point]
                insample_time_stamps[i, -len(insample_window_time_stamps):] = self.time_features(insample_window_time_stamps)
                outsample_window_timestamps = sampled_timestamps[
                                   cut_point:min(len(sampled_timestamps), cut_point + self.outsample_size)]
                outsample_time_stamps[i, :len(outsample_window_timestamps)] = self.time_features(outsample_window_timestamps)
            yield insample, insample_mask, outsample, outsample_mask, insample_time_stamps, outsample_time_stamps

    def sequential_latest_insamples(self):
        batch_size = len(self.timeseries)
        insample = np.zeros((batch_size, self.insample_size))
        insample_mask = np.zeros((batch_size, self.insample_size))
        insample_time_stamps = np.zeros(
                (batch_size, self.insample_size, self.time_embedding_dim), dtype=object)        
        for i, (ts, time_stamp) in enumerate(zip(self.timeseries, self.time_stamps)):
            ts_last_window = ts[-self.insample_size:]
            insample[i, -len(ts):] = ts_last_window
            insample_mask[i, -len(ts):] = 1.0

            sampled_timestamps = time_stamp
            insample_window_time_stamps = sampled_timestamps[-self.insample_size:]
            insample_time_stamps[i, -len(insample_window_time_stamps):] = self.time_features(insample_window_time_stamps)

        return insample, insample_mask, insample_time_stamps
