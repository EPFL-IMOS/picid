import random
from typing import Callable
from typing import Optional, Tuple

import torch
from torch import nn

from picid.model.definitions import CLASSIFICATION_TASKS, REGRESSION_TASKS
from picid.model.forecasters.forecaster import Forecaster
from picid.model.forecasters.spacetimeformer_model.nn.time2vec import Time2Vec


class LSTM_Encoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 256,
        n_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True
        )

    def forward(self, x_context: torch.Tensor):
        outputs, (hidden, cell) = self.lstm(x_context)
        return hidden, cell


class LSTM_Decoder(nn.Module):
    def __init__(
        self,
        output_dim: int = 1,
        input_dim: int = 1,
        hidden_dim: int = 256,
        n_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x_t, hidden, cell):
        output, (hidden, cell) = self.lstm(x_t, (hidden, cell))
        y_t1 = self.fc(output)
        return y_t1, hidden, cell


class LSTM_Seq2Seq(nn.Module):
    def __init__(self, t2v: Time2Vec, encoder: LSTM_Encoder, decoder: LSTM_Decoder):
        super().__init__()
        self.t2v = t2v
        self.encoder = encoder
        self.decoder = decoder

    def _merge(self, x, y):
        return torch.cat((x, y), dim=-1)

    def forward(
        self,
        x_context,
        y_context,
        x_target,
        y_target,
        teacher_forcing_prob,
    ):
        if self.t2v is not None:
            x_context = self.t2v(x_context)
            x_target = self.t2v(x_target)

        pred_len = y_target.shape[1]
        batch_size = y_target.shape[0]
        y_dim = y_target.shape[2]
        outputs = -torch.ones(batch_size, pred_len, y_dim).to(y_target.device)

        if x_context is not None:
            merged_context = self._merge(x_context, y_context)
            decoder_sequence = self._merge(
                x_context[:, -1], torch.zeros_like(y_target[:, 0])
            )
        else:
            decoder_sequence = torch.zeros_like(y_target[:, 0])
            merged_context = y_context

        hidden, cell = self.encoder(merged_context)

        decoder_input = decoder_sequence.unsqueeze(1).to(y_context.device)

        for t in range(0, pred_len):
            output, hidden, cell = self.decoder(decoder_input, hidden, cell)
            outputs[:, t] = output.squeeze(1)

            decoder_y = (
                y_target[:, t].unsqueeze(1)
                if random.random() < teacher_forcing_prob
                else output
            )
            if x_target is not None:
                decoder_input = self._merge(x_target[:, t].unsqueeze(1), decoder_y)
            else:
                decoder_input = decoder_y
        return outputs


class LSTM_Regression(nn.Module):
    def __init__(
        self,
        t2v: Time2Vec,
        encoder: LSTM_Encoder,
        decoder: LSTM_Decoder,
        n_classes: Optional[int] = None,
    ):
        super().__init__()
        self.t2v = t2v
        self.encoder = encoder
        self.decoder = decoder
        self.n_classes = n_classes

    def _merge(self, x, y):
        return torch.cat((x, y), dim=-1)

    def forward(
        self,
        x_target,
        y_target,
        teacher_forcing_prob,
    ):
        if self.t2v is not None:
            x_target = self.t2v(x_target)

        pred_len = y_target.shape[1]
        batch_size = y_target.shape[0]
        y_dim = y_target.shape[2] if self.n_classes is None else self.n_classes
        outputs = -torch.ones(batch_size, pred_len, y_dim).to(y_target.device)
        hidden, cell = self.encoder(x_target)

        decoder_input = x_target[:, -1].unsqueeze(1).to(x_target.device)

        for t in range(0, pred_len):
            output, hidden, cell = self.decoder(decoder_input, hidden, cell)
            outputs[:, t] = output.squeeze(1)

            decoder_y = (
                y_target[:, t].unsqueeze(1)
                if random.random() < teacher_forcing_prob
                else output
            )
            decoder_input = self._merge(x_target[:, t].unsqueeze(1), decoder_y)
        return outputs


class LSTM_Forecaster(Forecaster):
    def __init__(
        self,
        optimizer_factory: Callable[..., torch.optim.Optimizer],
        scheduler_factory: Callable[..., torch.optim.lr_scheduler._LRScheduler],
        d_x: int,
        d_yc: int,
        d_yt: int,
        task_type: str,
        time_emb_dim: int = 0,
        n_layers: int = 2,
        hidden_dim: int = 32,
        dropout_p: float = 0.2,
        # training
        teacher_forcing_prob: float = 0.5,
        loss: str = "mse",
        linear_window: int = 0,
        linear_shared_weights: bool = False,
        use_revin: bool = False,
        use_seasonal_decomp: bool = False,
        mask_y_c: bool = False,
        **kwargs,
    ):
        super().__init__(
            d_x=d_x,
            d_yc=d_yc,
            d_yt=d_yt,
            optimizer_factory=optimizer_factory,
            scheduler_factory=scheduler_factory,
            loss=loss,
            linear_window=linear_window,
            use_revin=use_revin,
            use_seasonal_decomp=use_seasonal_decomp,
            linear_shared_weights=linear_shared_weights,
            task_type=task_type,
            **kwargs,
        )

        if d_x > 0:
            self.t2v = Time2Vec(input_dim=d_x, embed_dim=time_emb_dim * d_x)
        else:
            self.t2v = None

        time_dim = time_emb_dim * d_x if time_emb_dim > 0 else d_x
        no_y_as_input = (
            task_type in REGRESSION_TASKS or task_type in CLASSIFICATION_TASKS
        )

        self.encoder = LSTM_Encoder(
            input_dim=time_dim + (d_yc if not no_y_as_input else 0),
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout_p,
        )
        self.decoder = LSTM_Decoder(
            output_dim=d_yt,
            input_dim=time_dim + (d_yt if not no_y_as_input else 0),
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout_p,
        )

        if self.task_type in REGRESSION_TASKS:
            self.model = LSTM_Regression(self.t2v, self.encoder, self.decoder).to(
                self.device
            )
        elif self.task_type in CLASSIFICATION_TASKS:
            assert d_yc == 0, "LSTM classifier does not support context labels"
            self.model = LSTM_Regression(
                self.t2v, self.encoder, self.decoder, n_classes=d_yt
            ).to(self.device)
        elif self.task_type in ["state_forecasting"]:
            self.model = LSTM_Seq2Seq(self.t2v, self.encoder, self.decoder).to(
                self.device
            )
        elif self.task_type in ["forecasting"]:
            self.model = LSTM_Seq2Seq(self.t2v, self.encoder, self.decoder).to(
                self.device
            )
        else:
            raise ValueError(f"Unknown task type {self.task_type}")

        self.teacher_forcing_prob = teacher_forcing_prob
        self.mask_y_c = mask_y_c

    @property
    def train_step_forward_kwargs(self):
        return {"force": self.teacher_forcing_prob}

    @property
    def eval_step_forward_kwargs(self):
        return {"force": 0.0}

    def forward_model_pass(self, x_c, y_c, x_t, y_t, force=None):
        if self.mask_y_c:
            y_c = torch.zeros_like(y_c)

        assert force is not None
        with torch.no_grad():
            # need to normalize y_t in LSTM because it is sometimes used
            # as input (teacher forcing). important to not leak the target
            # stats (update_stats = False).
            y_t = self.revin(y_t, mode="norm", update_stats=False)

        if self.task_type in REGRESSION_TASKS or self.task_type in CLASSIFICATION_TASKS:
            preds = self.model.forward(x_t, y_t, teacher_forcing_prob=force)
        elif self.task_type in ["forecasting", "state_forecasting"]:
            preds = self.model.forward(x_c, y_c, x_t, y_t, teacher_forcing_prob=force)
        return (preds,)

    def step(self, batch: Tuple[torch.Tensor], train: bool = False):
        stats = super().step(batch, train)
        stats["forecast_loss"] = stats["loss"]
        return stats
