from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange, repeat

from .time2vec import Time2Vec

# from layers.Embed_iTransformer import DataEmbedding_inverted
from .extra_layers import ConvBlock, Flatten

# import models as stf


class Embedding(nn.Module):
    def __init__(
        self,
        d_y,
        d_x,
        d_model,
        time_emb_dim=6,
        method="spatio-temporal",
        downsample_convs=0,
        start_token_len=0,
        null_value=None,
        pad_value=None,
        is_encoder: bool = True,
        variable_emb=None,
        position_emb="abs",
        timetable_emb="abs",  # "t2v" or "abs
        time_emb="t2v",  # "t2v" or "abs"
        embedding_router: Optional[torch.Tensor] = None,
        data_dropout=None,
        max_seq_len=None,
        use_val: bool = True,
        use_time: bool = True,
        use_timetable: bool = True,
        use_space: bool = True,
        use_given: bool = True,
    ):
        super().__init__()
        assert position_emb in ["t2v", "abs"]
        assert timetable_emb in ["t2v", "linear", None]
        assert time_emb in ["t2v", "linear"]
        assert method in ["spatio-temporal", "temporal", "inverted"]
        if embedding_router is not None:
            assert method == "spatio-temporal"

        if data_dropout is None:
            self.data_drop = lambda y: y
        else:
            self.data_drop = data_dropout

        self.method = method

        self.embedding_router = embedding_router
        if embedding_router is not None:
            if timetable_emb is None:
                # this can happen if we use the roter for negative start_token_len
                # not visible here because we set it to 0 in the model
                timetable_dim = 0
                self.timetable_emb = None
                d_x = int((torch.tensor(embedding_router) == 1).sum().item())

            else:
                self.mask_tt = torch.tensor(embedding_router) == 0
                self.mask_t = torch.tensor(embedding_router) == 1
                d_x_tt = self.mask_tt.sum()  # timetable_dim
                d_x = self.mask_t.sum()  # time_dim
                timetable_dim = time_emb_dim * d_x_tt

                if timetable_emb == "t2v":
                    self.timetable_emb = Time2Vec(d_x_tt, embed_dim=timetable_dim)
                elif timetable_emb == "linear":
                    self.timetable_emb = nn.Linear(
                        in_features=d_x_tt, out_features=timetable_dim
                    )
        else:
            timetable_dim = 0

        self.max_seq_len = max_seq_len
        self.position_emb = position_emb
        if self.position_emb == "t2v":
            # standard periodic pos emb but w/ learnable coeffs
            self.local_emb = Time2Vec(1, embed_dim=d_model + 1)
        elif self.position_emb == "abs":
            # lookup-based learnable pos emb
            assert max_seq_len is not None
            self.local_emb = nn.Embedding(
                num_embeddings=max_seq_len, embedding_dim=d_model
            )

        time_dim = time_emb_dim * d_x if (self.method != "inverted") else d_model
        if time_emb == "t2v":
            self.time_emb = Time2Vec(d_x, embed_dim=time_dim)
        elif time_emb == "linear":
            self.time_emb = nn.Linear(in_features=d_x, out_features=time_dim)

        y_emb_inp_dim = (
            d_y if ((self.method == "temporal") | (self.method == "inverted")) else 1
        )
        if variable_emb == "t2v":
            self.variable_emb = Time2Vec(y_emb_inp_dim, embed_dim=d_model)
            y_emb_inp_dim = d_model
        elif variable_emb == "linear":
            self.variable_emb = nn.Linear(
                in_features=y_emb_inp_dim, out_features=d_model
            )
            y_emb_inp_dim = d_model
        else:
            self.variable_emb = None

        if self.method != "inverted":
            self.val_time_emb = nn.Linear(
                y_emb_inp_dim + time_dim + timetable_dim, d_model
            )

        if self.method == "spatio-temporal":
            self.space_emb = nn.Embedding(num_embeddings=d_y, embedding_dim=d_model)
            split_length_into = d_y
        else:
            split_length_into = 1

        self.start_token_len = start_token_len
        self.given_emb = nn.Embedding(num_embeddings=2, embedding_dim=d_model)

        self.downsize_convs = nn.ModuleList(
            [ConvBlock(split_length_into, d_model) for _ in range(downsample_convs)]
        )

        self.d_model = d_model
        self.null_value = null_value
        self.pad_value = pad_value
        self.is_encoder = is_encoder

        # turning off parts of the embedding is only really here for ablation studies
        self.use_val = use_val
        self.use_time = use_time
        self.use_given = use_given
        self.use_space = use_space
        self.use_timetable = use_timetable

    def __call__(self, x: torch.Tensor, y: torch.Tensor):
        if self.method == "spatio-temporal":
            emb = self.spatio_temporal_embed
        elif self.method == "inverted":
            y = rearrange(y, "b l n -> b n l")
            x = rearrange(x, "b l n -> b n l")
            emb = self.temporal_embed
        elif self.method == "temporal":
            emb = self.temporal_embed
        return emb(y=y, x=x)

    def make_mask(self, y):
        # we make padding-based masks here due to outdated
        # feature where the embedding randomly drops tokens by setting
        # them to the pad value as a form of regularization
        if self.pad_value is None:
            return None
        return (y == self.pad_value).any(-1, keepdim=True)

    def temporal_embed(self, y: torch.Tensor, x: torch.Tensor):
        bs, length, d_y = y.shape

        # protect against true NaNs. without
        # `spatio_temporal_embed`'s multivariate "Given"
        # concept there isn't much else we can do here.
        # NaNs should probably be set to a magic number value
        # in the dataset and passed to the null_value arg.
        y = torch.nan_to_num(y)
        x = torch.nan_to_num(x)

        # keep track of pre-dropout y for given emb
        y_original = y.clone()
        if self.is_encoder:
            # optionally mask the context sequence for reconstruction
            y = self.data_drop(y)
        mask = self.make_mask(y)

        # position embedding ("local_emb")
        local_pos = torch.arange(length).to(x.device)
        if self.position_emb == "t2v":
            # first idx of Time2Vec output is unbounded so we drop it to
            # reuse code as a learnable pos embb
            local_emb = self.local_emb(
                local_pos.view(1, -1, 1).repeat(bs, 1, 1).float()
            )[:, :, 1:]
        elif self.position_emb == "abs":
            assert length <= self.max_seq_len
            local_emb = self.local_emb(local_pos.long().view(1, -1).repeat(bs, 1))

        # time embedding (Time2Vec)
        if not self.use_time:
            x = torch.zeros_like(x)
        time_emb = self.time_emb(x)

        if self.embedding_router is not None and self.timetable_emb is not None:
            x_tt = x[:, :, self.mask_tt]
            x = x[:, :, self.mask_t]
            timetable_emb = self.timetable_emb(x_tt)
        else:
            timetable_emb = None

        if not self.use_val:
            y = torch.zeros_like(y)

        # for use_given, need to be computed before variable emb
        _y_dropped = (y == y_original).squeeze(-1)
        if self.variable_emb is not None:
            y = self.variable_emb(y)

        # concat time_emb, y --> FF --> val_time_emb
        embed_dim_idx = 1 if self.method == "inverted" else -1
        if timetable_emb is not None:
            val_time_inp = torch.cat((time_emb, timetable_emb, y), dim=embed_dim_idx)
        else:
            val_time_inp = torch.cat((time_emb, y), dim=embed_dim_idx)

        if self.method != "inverted":
            val_time_emb = self.val_time_emb(val_time_inp)

            # "given" embedding. not important for temporal emb
            # when not using a start token
            given = torch.ones((bs, length)).long().to(x.device)
            if not self.is_encoder and self.use_given:
                given[:, self.start_token_len :] = 0
            given_emb = self.given_emb(given)

            emb = local_emb + val_time_emb + given_emb
        else:
            emb = val_time_inp

        if self.is_encoder:
            # shorten the sequence
            for i, conv in enumerate(self.downsize_convs):
                emb = conv(emb)

        # space emb not used for temporal method
        space_emb = torch.zeros_like(emb)
        var_idxs = None
        return emb, space_emb, var_idxs, mask

    def spatio_temporal_embed(self, y: torch.Tensor, x: torch.Tensor):
        # full spatiotemopral emb method. lots of shape rearrange code
        # here to create artifically long (length x dim) spatiotemporal sequence
        batch, length, dy = y.shape

        # position emb ("local_emb")
        local_pos = repeat(
            torch.arange(length).to(x.device), f"length -> {batch} ({dy} length)"
        )
        if self.position_emb == "t2v":
            # periodic pos emb
            local_emb = self.local_emb(local_pos.float().unsqueeze(-1).float())[
                :, :, 1:
            ]
        elif self.position_emb == "abs":
            # lookup pos emb
            local_emb = self.local_emb(local_pos.long())

        # time emb
        if not self.use_time:
            x = torch.zeros_like(x)
        x = torch.nan_to_num(x)
        x = repeat(x, f"batch len x_dim -> batch ({dy} len) x_dim")

        if self.embedding_router is not None and self.timetable_emb is not None:
            x_tt = x[:, :, self.mask_tt]
            x = x[:, :, self.mask_t]

            if not self.use_timetable:
                x_tt = torch.zeros_like(x_tt)

            timetable_emb = self.timetable_emb(x_tt)
        else:
            timetable_emb = None

        time_emb = self.time_emb(x)

        # protect against NaNs in y, but keep track for Given emb
        true_null = torch.isnan(y)
        y = torch.nan_to_num(y)
        if not self.use_val:
            y = torch.zeros_like(y)

        # keep track of pre-dropout y for given emb
        y_original = y.clone()
        y_original = Flatten(y_original)
        y = self.data_drop(y)
        y = Flatten(y)
        mask = self.make_mask(y)

        # for use_given, need to be computed before variable emb
        y_dropped = (y == y_original).squeeze(-1)
        if self.null_value is not None:
            y_null = (y != self.null_value).squeeze(-1)

        if self.variable_emb is not None:
            y = self.variable_emb(y)

        # concat time_emb, y --> FF --> val_time_emb
        if timetable_emb is not None:
            val_time_inp = torch.cat((time_emb, timetable_emb, y), dim=-1)
        else:
            val_time_inp = torch.cat((time_emb, y), dim=-1)

        val_time_emb = self.val_time_emb(val_time_inp)

        # "given" embedding
        if self.use_given:
            given = torch.ones((batch, length, dy)).long().to(x.device)  # start as True
            if not self.is_encoder:
                # mask missing values that need prediction...
                given[:, self.start_token_len :, :] = 0  # (False)

            # if y was NaN, set Given = False
            given *= ~true_null

            # flatten now to make the rest easier to figure out
            given = rearrange(given, "batch len dy -> batch (dy len)")

            # use given embeddings to identify data that was dropped out
            given *= y_dropped

            if self.null_value is not None:
                # mask null values that were set to a magic number in the dataset itself
                null_mask = y_null
                given *= null_mask

            given_emb = self.given_emb(given)
        else:
            given_emb = 0.0

        val_time_emb = local_emb + val_time_emb + given_emb

        if self.is_encoder:
            for conv in self.downsize_convs:
                val_time_emb = conv(val_time_emb)
                length //= 2

        # space embedding
        var_idx = repeat(
            torch.arange(dy).long().to(x.device), f"dy -> {batch} (dy {length})"
        )
        var_idx_true = var_idx.clone()
        if not self.use_space:
            var_idx = torch.zeros_like(var_idx)
        space_emb = self.space_emb(var_idx)

        return val_time_emb, space_emb, var_idx_true, mask


# TODO: rows 353-366 are potentially dead code. Reason: Commented-out IEmbedding class never executed.
# Additional Note: Remove if confirmed unused.
# class IEmbedding(nn.Module):
#     def __init__(self, c_in, d_model, dropout=0.1):
#         super().__init__()
#         self.inverted_emb = DataEmbedding_inverted(
#             c_in=c_in, d_model=d_model, dropout=dropout
#         )

#     def forward(self, y, x):
#         val_time_emb = self.inverted_emb(x=y, x_mark=x)

#         # space emb not used for temporal method
#         space_emb = torch.zeros_like(val_time_emb)
#         mask = None
#         var_idx_true = None
#         return val_time_emb, space_emb, var_idx_true, mask
#         return val_time_emb, space_emb, var_idx_true, mask
