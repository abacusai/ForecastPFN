import warnings

import pmdarima
import torch

from exp.exp_basic import Exp_Basic

warnings.filterwarnings('ignore')


class Exp_SeasonalNaive(Exp_Basic):
    def __init__(self, args):
        super(Exp_SeasonalNaive, self).__init__(args)

    def _build_model(self):
        return pmdarima.auto_arima

    def train(self, setting):
        pass

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')

        preds, trues = [], []
        self.test_timer.start_timer()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                test_loader
            ):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                true = batch_y[:, -self.args.pred_len :].detach().cpu().numpy()
                pred = (
                    batch_x[:, -7:, :]
                    .repeat(1, int(true.shape[1] / 7) + 1, 1)[:, : true.shape[1], :]
                    .detach()
                    .cpu()
                    .numpy()
                )

                preds.append(pred)
                trues.append(true)
        self.test_timer.end_timer()

        return self._save_test_data(setting, preds, trues)
