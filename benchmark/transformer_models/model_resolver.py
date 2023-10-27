import pandas as pd
import numpy as np
import prophet
import pmdarima

from transformer_models.models import FEDformer, Autoformer, Informer, Transformer


class Arima():
    def __init__(self) -> None:
        self.model = pmdarima.auto_arima


class Prophet():
    def __init__(self) -> None:
        self.model = prophet.Prophet()


model_dict = {
    'FEDformer': FEDformer,
    'FEDformer-w': FEDformer,
    'FEDformer-f': FEDformer,
    'Autoformer': Autoformer,
    'Transformer': Transformer,
    'Informer': Informer,
    'Arima': Arima,
    'Prophet': Prophet,
}
