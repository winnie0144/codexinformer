import math
import os
import sys

import torch
import torch.nn as nn

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
GITINFORMER_DIR = os.path.join(PROJECT_DIR, "gitinformer")
if GITINFORMER_DIR not in sys.path:
    sys.path.insert(0, GITINFORMER_DIR)

from models.attn import ProbAttention, AttentionLayer, FullAttention
from models.encoder import Encoder, EncoderLayer, ConvLayer
from models.decoder import Decoder, DecoderLayer


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.pe[:, : x.size(1)]


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.linear = nn.Linear(4, d_model)

    def forward(self, x_mark):
        return self.linear(x_mark.float())


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super().__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.position_embedding = PositionalEmbedding(d_model)
        self.temporal_embedding = TemporalEmbedding(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x_value, x_mark):
        x = (
            self.value_embedding(x_value)
            + self.position_embedding(x_value)
            + self.temporal_embedding(x_mark)
        )
        return self.dropout(x)


class InformerSAIF(nn.Module):
    def __init__(
        self,
        enc_in,
        dec_in,
        out_len,
        d_model=512,
        nhead=16,
        d_ff=2048,
        dropout=0.1,
        factor=5,
        e_layers=3,
        d_layers=2,
    ):
        super().__init__()
        self.pred_len = out_len

        self.enc_embedding = DataEmbedding(enc_in, d_model, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, dropout)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        ProbAttention(False, factor, attention_dropout=dropout),
                        d_model,
                        nhead,
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation="relu",
                )
                for _ in range(e_layers)
            ],
            [ConvLayer(d_model) for _ in range(e_layers - 1)],
            norm_layer=torch.nn.LayerNorm(d_model),
        )

        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        FullAttention(True, factor, attention_dropout=dropout),
                        d_model,
                        nhead,
                    ),
                    AttentionLayer(
                        FullAttention(False, factor, attention_dropout=dropout),
                        d_model,
                        nhead,
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation="relu",
                )
                for _ in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model),
        )

        self.projection = nn.Linear(d_model, 2)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        dec_out = self.dec_embedding(x_dec, x_mark_dec)

        enc_out, _ = self.encoder(enc_out, attn_mask=None)
        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)
        f_out = self.projection(dec_out[:, -self.pred_len :, :])

        mu = f_out[:, :, 0]
        log_var = torch.clamp(f_out[:, :, 1], min=-5.0, max=3.0)
        return mu, log_var
