from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import torchmetrics

import pkg_resources

from lmetk.lr_scheduler import WarmupReduceLROnPlateauScheduler
from .nn import Spacetimeformer

from lmetk.baselines.forecaster import Forecaster


def find_ckpt(path):
    path = path / "checkpoints"
    files = [
        x
        for x in path.iterdir()
        if x.is_file() and x.suffix == ".ckpt" and x.stem != "last"
    ]
    return files[0]


class Spacetimeformer_Forecaster(Forecaster):
    def __init__(
        self,
        d_yc: int = 1,
        d_yt: int = 1,
        d_x: int = 4,
        max_seq_len: int = None,
        start_token_len: int = 64,
        attn_factor: int = 5,
        d_model: int = 200,
        d_queries_keys=50,
        d_values=50,
        n_heads: int = 4,
        e_layers: int = 2,
        d_layers: int = 2,
        d_ff: int = 800,
        dropout_emb: float = 0.1,
        dropout_qkv: float = 0.0,
        dropout_ff: float = 0.2,
        dropout_attn_out: float = 0.0,
        dropout_attn_matrix: float = 0.0,
        pos_emb_type: str = "abs",
        timetable_emb_type: Optional[str] = None,
        time_emb_type: str = "t2v",
        variable_emb_type: Optional[str] = None,
        inverted_encoder: bool = False,  # iTransformer style encoder
        global_self_attn: str = "performer",
        local_self_attn: str = "performer",
        global_cross_attn: str = "performer",
        local_cross_attn: str = "performer",
        performer_kernel: str = "relu",
        embed_method: str = "spatio-temporal",
        attention_method: str = "non-causual",
        d_x_embedding_router: Optional[torch.Tensor] = None,
        performer_relu: bool = True,
        performer_redraw_interval: int = 1000,
        attn_time_windows: int = 1,
        use_shifted_time_windows: bool = True,
        activation: str = "gelu",
        head_f: str = "linear",
        norm: str = "batch",
        use_final_norm: bool = True,
        init_lr: float = 1e-10,
        base_lr: float = 5e-4,
        patience: int = 3,
        warmup_steps: float = 1000,
        decay_factor: float = 0.8,
        initial_downsample_convs: int = 0,
        intermediate_downsample_convs: int = 0,
        l2_coeff: float = 1e-3,
        loss: str = "mse",
        class_loss_imp: float = 1e-3,
        recon_loss_imp: float = 0,
        time_emb_dim: int = 6,
        null_value: float = None,
        pad_value: float = None,
        linear_window: int = 0,
        linear_shared_weights: bool = False,
        use_revin: bool = False,
        use_seasonal_decomp: bool = False,
        use_val_enc: bool = True,
        use_val_dec: bool = True,
        use_tt_enc: bool = True,
        use_tt_dec: bool = True,
        use_time: bool = True,
        use_space: bool = True,
        use_given: bool = True,
        recon_mask_skip_all: float = 1.0,
        recon_mask_max_seq_len: int = 5,
        recon_mask_drop_seq: float = 0.1,
        recon_mask_drop_standard: float = 0.2,
        recon_mask_drop_full: float = 0.05,
        verbose=True,
        target_mask=None,
        **kwargs
    ):
        super().__init__(
            d_x=d_x,
            d_yc=d_yc,
            d_yt=d_yt,
            l2_coeff=l2_coeff,
            loss=loss,
            linear_window=linear_window,
            use_revin=use_revin,
            use_seasonal_decomp=use_seasonal_decomp,
            linear_shared_weights=linear_shared_weights,
        )

        self.torchmetrics_version = pkg_resources.get_distribution(
            "torchmetrics"
        ).version

        if start_token_len < 0:
            timetable_emb_enc_type = timetable_emb_type
            timetable_emb_dec_type = None
        else:
            timetable_emb_enc_type = timetable_emb_type
            timetable_emb_dec_type = timetable_emb_type

        self.spacetimeformer = Spacetimeformer(
            d_yc=d_yc,
            d_yt=d_yt,
            d_x=d_x,
            start_token_len=start_token_len if start_token_len >= 0 else 0,
            attn_factor=attn_factor,
            d_model=d_model,
            d_queries_keys=d_queries_keys,
            d_values=d_values,
            n_heads=n_heads,
            e_layers=e_layers,
            d_layers=d_layers,
            d_ff=d_ff,
            initial_downsample_convs=initial_downsample_convs,
            intermediate_downsample_convs=intermediate_downsample_convs,
            dropout_emb=dropout_emb,
            dropout_attn_out=dropout_attn_out,
            dropout_attn_matrix=dropout_attn_matrix,
            dropout_qkv=dropout_qkv,
            dropout_ff=dropout_ff,
            pos_emb_type=pos_emb_type,
            timetable_emb_enc_type=timetable_emb_enc_type,
            timetable_emb_dec_type=timetable_emb_dec_type,
            inverted_encoder=inverted_encoder,
            time_emb_type=time_emb_type,
            variable_emb_type=variable_emb_type,
            global_self_attn=global_self_attn,
            local_self_attn=local_self_attn,
            global_cross_attn=global_cross_attn,
            local_cross_attn=local_cross_attn,
            activation=activation,
            head_f=head_f,
            device=self.device,
            norm=norm,
            use_final_norm=use_final_norm,
            embed_method=embed_method,
            attention_method=attention_method,
            d_x_embedding_router=d_x_embedding_router,
            performer_attn_kernel=performer_kernel,
            performer_redraw_interval=performer_redraw_interval,
            attn_time_windows=attn_time_windows,
            use_shifted_time_windows=use_shifted_time_windows,
            time_emb_dim=time_emb_dim,
            verbose=True,
            null_value=null_value,
            pad_value=pad_value,
            max_seq_len=max_seq_len,
            use_val_enc=use_val_enc,
            use_val_dec=use_val_dec,
            use_tt_dec=use_tt_dec,
            use_tt_enc=use_tt_enc,
            use_time=use_time,
            use_space=use_space,
            use_given=use_given,
            recon_mask_skip_all=recon_mask_skip_all,
            recon_mask_max_seq_len=recon_mask_max_seq_len,
            recon_mask_drop_seq=recon_mask_drop_seq,
            recon_mask_drop_standard=recon_mask_drop_standard,
            recon_mask_drop_full=recon_mask_drop_full,
        )

        self.target_mask = target_mask
        self.start_token_len = start_token_len
        self.init_lr = init_lr
        self.base_lr = base_lr
        self.patience = patience
        self.warmup_steps = warmup_steps
        self.decay_factor = decay_factor
        self.embed_method = embed_method
        self.class_loss_imp = class_loss_imp
        self.recon_loss_imp = recon_loss_imp
        self.set_null_value(null_value)
        self.pad_value = pad_value
        self.d_x_embedding_router = (
            torch.tensor(d_x_embedding_router).to(self.device)
            if d_x_embedding_router is not None
            else None
        )
        self.save_hyperparameters()

        self.validation_step_outputs = []

    @property
    def train_step_forward_kwargs(self):
        kwargs = {"output_attn": False}
        if self.target_mask is not None:
            kwargs["target_mask"] = self.target_mask
        return kwargs

    @property
    def eval_step_forward_kwargs(self):
        kwargs = {"output_attn": False}
        if self.target_mask is not None:
            kwargs["target_mask"] = self.target_mask
        return kwargs

    def step(self, batch: Tuple[torch.Tensor], train: bool):

        # x_c, y_c, x_t, y_t, 
        batch = (
            batch["context"]["x"],
            batch["target"]["x"],
            batch["context"]["y"],
            batch["target"]["y"]
        )

        kwargs = (
            self.train_step_forward_kwargs if train else self.eval_step_forward_kwargs
        )

        time_mask = self.time_masked_idx if train else None

        loss_dict = self.compute_loss(
            batch=batch,
            time_mask=time_mask,
            forward_kwargs=kwargs,
        )

        forecast_out = loss_dict["forecast_out"]
        forecast_mask = loss_dict["forecast_mask"]
        *_, y_t = batch

        stats = self._compute_stats(forecast_out, y_t, forecast_mask)

        stats["forecast_loss"] = loss_dict["forecast_loss"]
        stats["class_loss"] = loss_dict["class_loss"]
        stats["recon_loss"] = loss_dict["recon_loss"]

        stats["loss"] = (
            loss_dict["forecast_loss"]
            + self.class_loss_imp * loss_dict["class_loss"]
            + self.recon_loss_imp * loss_dict["recon_loss"]
        )
        stats["acc"] = loss_dict["acc"]
        return stats

    def classification_loss(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[torch.Tensor]:
        labels = labels.view(-1).to(logits.device)
        d_y = labels.max() + 1

        logits = logits.view(-1, d_y)

        class_loss = F.cross_entropy(logits, labels)

        if self.torchmetrics_version == "0.9.3":
            acc = torchmetrics.functional.accuracy(
                torch.softmax(logits, dim=1),
                labels,
            )
        else:
            if d_y == 1:
                acc = torchmetrics.functional.accuracy(
                    torch.softmax(logits, dim=1).squeeze(1), labels, task="binary"
                )
            else:
                acc = torchmetrics.functional.accuracy(
                    torch.softmax(logits, dim=1),
                    labels,
                    task="multiclass",
                    num_classes=d_y.item(),
                )
        return class_loss, acc

    def compute_loss(self, batch, time_mask=None, forward_kwargs={}):
        x_c, y_c, x_t, y_t = batch

        target_mask = forward_kwargs.get("target_mask", None)

        forecast_out, recon_out, (logits, labels) = self(
            x_c, y_c, x_t, y_t, **forward_kwargs
        )

        if target_mask is not None:
            y_t = y_t[:, :, target_mask]

        forecast_loss, forecast_mask = self.forecasting_loss(
            outputs=forecast_out, y_t=y_t, time_mask=time_mask
        )

        if self.recon_loss_imp > 0:
            recon_loss, recon_mask = self.forecasting_loss(
                outputs=recon_out, y_t=y_c, time_mask=None
            )
        else:
            recon_loss, recon_mask = -1.0, 0.0

        if self.embed_method == "spatio-temporal" and self.class_loss_imp > 0:
            class_loss, acc = self.classification_loss(logits=logits, labels=labels)
        else:
            class_loss, acc = 0.0, -1.0

        return {
            "forecast_loss": forecast_loss,
            "class_loss": class_loss,
            "acc": acc,
            "forecast_out": forecast_out,
            "forecast_mask": forecast_mask,
            "recon_out": recon_out,
            "recon_loss": recon_loss,
            "recon_mask": recon_mask,
        }

    def nan_to_num(self, *inps):
        return inps

    def forward_model_pass(
        self, x_c, y_c, x_t, y_t, output_attn=False, target_mask=None
    ):
        if len(y_c.shape) == 2:
            y_c = y_c.unsqueeze(-1)
        if len(y_t.shape) == 2:
            y_t = y_t.unsqueeze(-1)

        enc_x = x_c
        enc_y = y_c
        dec_x = x_t

        if target_mask is not None:
            dec_y = y_t.clone()
            dec_y[:, :, target_mask] = 0.0
        else:
            dec_y = torch.zeros_like(y_t).to(self.device)

        if self.start_token_len > 0:
            dec_y = torch.cat((y_c[:, -self.start_token_len :, :], dec_y), dim=1).to(
                self.device
            )
            dec_x = torch.cat((x_c[:, -self.start_token_len :, :], dec_x), dim=1)
        if self.start_token_len < 0:
            enc_y = torch.cat((enc_y, dec_y[:, self.start_token_len :, :]), dim=1).to(
                self.device
            )
            enc_x = torch.cat((enc_x, dec_x[:, self.start_token_len :, :]), dim=1)

            dec_x = dec_x[:, :, self.d_x_embedding_router == 1]

        (
            forecast_output,
            recon_output,
            (logits, labels),
            attn,
            (enc_latent, dec_latent),
        ) = self.spacetimeformer(
            enc_x=enc_x,
            enc_y=enc_y,
            dec_x=dec_x,
            dec_y=dec_y,
            output_attention=output_attn,
        )

        if target_mask is not None:
            forecast_output = forecast_output[:, :, target_mask]

        if output_attn:
            return forecast_output, recon_output, (logits, labels), attn
        return forecast_output, recon_output, (logits, labels)

    def validation_step(self, batch, batch_idx):
        outs = super().validation_step(batch, batch_idx)
        self.validation_step_outputs.append(outs)
        return outs

    def on_validation_epoch_end(self):
        outs = self.validation_step_outputs
        total = 0
        count = 0
        for dict_ in outs:
            if "forecast_loss" in dict_:
                total += dict_["forecast_loss"].mean()
                count += 1
        avg_val_loss = total / count
        self.scheduler.step(avg_val_loss, is_end_epoch=True)
        self.validation_step_outputs.clear()

    def training_step(self, batch, batch_idx):
        outs = super().training_step(batch, batch_idx)
        self.scheduler.step()
        return outs

    def configure_optimizers(self):
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.base_lr,
            weight_decay=self.l2_coeff,
        )
        self.scheduler = WarmupReduceLROnPlateauScheduler(
            self.optimizer,
            init_lr=self.init_lr,
            peak_lr=self.base_lr,
            warmup_steps=self.warmup_steps,
            patience=self.patience,
            factor=self.decay_factor,
        )
        # return [self.optimizer], [self.scheduler]
        return dict(
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            monitor="loss"
        )
