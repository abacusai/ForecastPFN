import numpy as np


def smape_m3(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Calculate sMAPE.
    This is the metric that is used in M3, http://www.forecastingprinciples.com/files/pdf/Makridakia-The%20M3%20Competition.pdf, Appendix A
    This method accepts one or many timeseries.
    For multiple timeseries pass matrix (N, M) where N is number of timeseries and M is number of time steps.

    :param prediction: Predicted values.
    :param target: Target values.
    :return: Same shape array with sMAPE calculated for each time step of each timeseries.
    """
    return 200 * np.abs(prediction - target) / (target + prediction)


def smape_m3_dataset_horizon(target_dataset, forecast_dataset, horizon):
    """
    Calculate sMAPE over all timeseries that have specified horizon.
    This is the metric that is used in M3, http://www.forecastingprinciples.com/files/pdf/Makridakia-The%20M3%20Competition.pdf, Appendix A
    This reproduces results in http://www.forecastingprinciples.com/files/pdf/Makridakia-The%20M3%20Competition.pdf,
    Table 6, Forecasting horizon for Naive2 and Single
    This method accepts M3Dataset.

    :param target_dataset: target values, M3Dataset.
    :param forecast_dataset: forecast values, M3Dataset.
    :return: scalar aggregated over all timeseries for horizon.
    """
    smape_cum = 0.0
    smape_n_points = 0.0
    i = 0
    for prediction, target in zip(forecast_dataset.values, target_dataset.values):
        if target_dataset.horizons[i] >= horizon:
            smape_cum += smape_m3(prediction[horizon-1], target[-len(prediction)+horizon-1]).sum()
            smape_n_points += 1
        i += 1

    return smape_cum / smape_n_points


def smape_m3_dataset_horizon_avg(target_dataset, forecast_dataset, horizon):
    """
    Calculate sMAPE for prediction and target over all timeseries, averaged between horizon=1 and specified horizon.
    This is the metric that is used in M3, http://www.forecastingprinciples.com/files/pdf/Makridakia-The%20M3%20Competition.pdf, Appendix A
    This reproduces results in http://www.forecastingprinciples.com/files/pdf/Makridakia-The%20M3%20Competition.pdf,
    Table 6, Average of forecasting horizon for Naive2 and Single
    This method accepts M3Dataset.

    :param target_dataset: target values, M3Dataset.
    :param forecast_dataset: forecast values, M3Dataset.
    :return: scalar aggregated over all timeseries for horizon.
    """
    smape_cum = 0.0
    smape_n_points = 0.0
    i = 0
    for prediction, target in zip(forecast_dataset.values, target_dataset.values):
        horizon_clamped = min(target_dataset.horizons[i], horizon)
        if horizon_clamped == target_dataset.horizons[i]:
            target_clamped = target[-target_dataset.horizons[i]:]
        else:
            target_clamped = target[-target_dataset.horizons[i]:-target_dataset.horizons[i]+horizon_clamped]
        smape_cum += smape_m3(prediction[:horizon_clamped], target_clamped).sum()
        smape_n_points += len(target_clamped)
        i += 1

    return smape_cum / smape_n_points


def get_masep(insample, freq):
    y_hat_naive = []
    for i in range(freq, len(insample)):
        y_hat_naive.append(insample[(i - freq)])

    masep = np.mean(abs(insample[freq:] - y_hat_naive))
    return masep


def smape_m3(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Calculate sMAPE for prediction and target.
    This is the metric that is used in M3, http://www.forecastingprinciples.com/files/pdf/Makridakia-The%20M3%20Competition.pdf, Appendix A
    This method accepts one or many timeseries.
    For multiple timeseries pass matrix (N, M) where N is number of timeseries and M is number of time steps.

    :param prediction: Predicted values.
    :param target: Target values.
    :return: Same shape array with sMAPE calculated for each time step of each timeseries.
    """
    return 200 * np.abs(prediction - target) / (target + prediction)


def smape_m3_dataset_horizon(target_dataset, forecast_dataset, horizon):
    """
    Calculate sMAPE for prediction and target over all timeseries that have specified horizon.
    This is the metric that is used in M3, http://www.forecastingprinciples.com/files/pdf/Makridakia-The%20M3%20Competition.pdf, Appendix A
    This reproduces results in http://www.forecastingprinciples.com/files/pdf/Makridakia-The%20M3%20Competition.pdf,
    Table 6, Forecasting horizon for Naive2 and Single
    This method accepts M3Dataset.

    :param target_dataset: target values, M3Dataset.
    :param forecast_dataset: forecast values, M3Dataset.
    :return: scalar aggregated over all timeseries for horizon.
    """
    smape_cum = 0.0
    smape_n_points = 0.0
    i = 0
    for prediction, target in zip(forecast_dataset.values, target_dataset.values):
        if target_dataset.horizons[i] >= horizon:
            smape_cum += smape_m3(prediction[horizon-1], target[-len(prediction)+horizon-1]).sum()
            smape_n_points += 1
        i += 1

    return smape_cum / smape_n_points


def smape_m3_dataset_horizon_avg(target_dataset, forecast_dataset, horizon):
    """
    Calculate sMAPE for prediction and target over all timeseries, averaged between horizon=1 and specified horizon.
    This is the metric that is used in M3, http://www.forecastingprinciples.com/files/pdf/Makridakia-The%20M3%20Competition.pdf, Appendix A
    This reproduces results in http://www.forecastingprinciples.com/files/pdf/Makridakia-The%20M3%20Competition.pdf,
    Table 6, Average of forecasting horizon for Naive2 and Single
    This method accepts M3Dataset.

    :param target_dataset: target values, M3Dataset.
    :param forecast_dataset: forecast values, M3Dataset.
    :return: scalar aggregated over all timeseries for horizon.
    """
    smape_cum = 0.0
    smape_n_points = 0.0
    i = 0
    for prediction, target in zip(forecast_dataset.values, target_dataset.values):
        horizon_clamped = min(target_dataset.horizons[i], horizon)
        if horizon_clamped == target_dataset.horizons[i]:
            target_clamped = target[-target_dataset.horizons[i]:]
        else:
            target_clamped = target[-target_dataset.horizons[i]:-target_dataset.horizons[i]+horizon_clamped]
        smape_cum += smape_m3(prediction[:horizon_clamped], target_clamped).sum()
        smape_n_points += len(target_clamped)
        i += 1

    return smape_cum / smape_n_points

