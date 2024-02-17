import pmdarima
import prophet

from transformer_models.models import Autoformer, FEDformer, Informer, Transformer


class Arima:
    def __init__(self) -> None:
        self.model = pmdarima.auto_arima


class Prophet:
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
