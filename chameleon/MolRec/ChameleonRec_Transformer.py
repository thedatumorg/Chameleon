import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

# ------------------------------------------------------------------------
# Conformer backbone
# ------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding.
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, D)
        """
        L = x.size(1)
        return x + self.pe[:, :L, :]


class ConformerConvModule(nn.Module):
    """
    Conformer convolution module.
    Input/Output: (B, L, D)
    """
    def __init__(self, d_model: int, kernel_size: int = 15):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.pointwise_conv1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=d_model,
        )
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.activation = nn.SiLU()
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, D)
        """
        x = self.layer_norm(x)
        x = x.transpose(1, 2)  # (B, D, L)
        x = self.pointwise_conv1(x)  # (B, 2D, L)
        x = self.glu(x)              # (B, D, L)
        x = self.depthwise_conv(x)   # (B, D, L)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pointwise_conv2(x)  # (B, D, L)
        x = x.transpose(1, 2)        # (B, L, D)
        return x


class ConformerBlock(nn.Module):
    """
    A single Conformer block:
      x <- x + 1/2 FFN(x)
      x <- x + MHSA(LN(x))
      x <- x + ConvModule(x)
      x <- x + 1/2 FFN(x)
      x <- LN(x)
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        ff_mult: int = 4,
        conv_kernel_size: int = 15,
        dropout: float = 0.1,
    ):
        super().__init__()
        d_ff = d_model * ff_mult

        # FFN 1
        self.ffn1_ln = nn.LayerNorm(d_model)
        self.ffn1 = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

        # MHSA
        self.mha_ln = nn.LayerNorm(d_model)
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Conv module
        self.conv_module = ConformerConvModule(d_model, kernel_size=conv_kernel_size)

        # FFN 2
        self.ffn2_ln = nn.LayerNorm(d_model)
        self.ffn2 = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

        # Final LayerNorm
        self.final_ln = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, src_mask=None, src_key_padding_mask=None) -> torch.Tensor:
        """
        x: (B, L, D)
        """
        # FFN half-step
        residual = x
        x_ln = self.ffn1_ln(x)
        x = residual + 0.5 * self.ffn1(x_ln)

        # MHSA
        residual = x
        x_ln = self.mha_ln(x)
        x_attn, _ = self.mha(
            x_ln, x_ln, x_ln,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=False,
        )
        x = residual + x_attn

        # Conv module
        residual = x
        x = residual + self.conv_module(x)

        # FFN second half-step
        residual = x
        x_ln = self.ffn2_ln(x)
        x = residual + 0.5 * self.ffn2(x_ln)

        # Final norm
        x = self.final_ln(x)
        return x


class TimeSeriesConformerEncoder(nn.Module):
    """
    Conformer encoder for time series: (B, C, L) -> (B, L, D)

    - Project channel dimension C -> d_model
    - Add positional encoding
    - Pass through N stacked Conformer blocks
    """
    def __init__(
        self,
        in_channels: int,
        d_model: int,
        num_layers: int = 4,
        n_heads: int = 4,
        ff_mult: int = 4,
        conv_kernel_size: int = 15,
        dropout: float = 0.1,
        max_len: int = 4096,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.d_model = d_model

        self.input_proj = nn.Linear(in_channels, d_model)
        self.pos_encoding = PositionalEncoding(d_model=d_model, max_len=max_len)

        self.layers = nn.ModuleList([
            ConformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                ff_mult=ff_mult,
                conv_kernel_size=conv_kernel_size,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, L) -> (B, L, D)
        """
        x = x.transpose(1, 2)           # (B, L, C)
        x = self.input_proj(x)          # (B, L, D)
        x = self.pos_encoding(x)        # (B, L, D)

        for layer in self.layers:
            x = layer(x)

        return x                        # (B, L, D)


# ------------------------------------------------------------------------
# Fusion + temporal pooling
# ------------------------------------------------------------------------
class GatedFusion(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.LayerNorm(2 * d_model),
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )
        self.proj = nn.Linear(2 * d_model, d_model)

    def forward(self, h_n, h_a):
        H = torch.cat([h_n, h_a], dim=-1)      # (B,L,2D)
        g = self.gate(H)                       # (B,L,D)
        mixed = g * h_a + (1 - g) * h_n        # (B,L,D)
        return self.proj(torch.cat([mixed, H[..., :H.shape[-1]//2]], dim=-1))


class ResidualAttentionPool(nn.Module):
    """
    Attention over time with residual energy as a prior.
    feats: (B,L',D)
    resid: (B,C,L)  -> pooled to (B,L')
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.score = nn.Linear(d_model, 1)

    def forward(self, feats, resid):
        B, Lp, D = feats.shape
        resid_energy = (resid**2).mean(dim=1)  # (B,L)
        resid_pooled = F.adaptive_avg_pool1d(
            resid_energy.unsqueeze(1), Lp
        ).squeeze(1)  # (B,L')
        resid_prior = (resid_pooled - resid_pooled.mean(dim=1, keepdim=True)) / \
                      (resid_pooled.std(dim=1, keepdim=True) + 1e-6)

        s = self.score(feats).squeeze(-1)  # (B,L')
        logits = s + resid_prior
        w = torch.softmax(logits, dim=1).unsqueeze(-1)  # (B,L',1)
        return (feats * w).sum(dim=1)  # (B,D)


# ------------------------------------------------------------------------
# Model with dual Conformer encoders on normal/residual streams
# (takes x_n, x_r as inputs; decomposition is done in the dataset)
# ------------------------------------------------------------------------
class ChameleonRec_Transformer(nn.Module):
    def __init__(
        self,
        in_channels,
        num_models,
        d_model: int = 256,
        num_layers: int = 4,
        n_heads: int = 4,
        ff_mult: int = 4,
        conv_kernel_size: int = 15,
        dropout: float = 0.1
    ):
        super().__init__()

        self.d_model = d_model

        # Two Conformer branches (normal / anomaly residual)
        self.encoder_normal = TimeSeriesConformerEncoder(
            in_channels=in_channels,
            d_model=d_model,
            num_layers=num_layers,
            n_heads=n_heads,
            ff_mult=ff_mult,
            conv_kernel_size=conv_kernel_size,
            dropout=dropout,
        )

        self.encoder_anomaly = TimeSeriesConformerEncoder(
            in_channels=in_channels,
            d_model=d_model,
            num_layers=num_layers,
            n_heads=n_heads,
            ff_mult=ff_mult,
            conv_kernel_size=conv_kernel_size,
            dropout=dropout,
        )

        self.fusion = GatedFusion(d_model)
        self.pool = ResidualAttentionPool(d_model)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_models),
        )

    def forward(self, x_n, x_r):  # both (B,C,L)
        h_n = self.encoder_normal(x_n)   # (B,L',D)
        h_r = self.encoder_anomaly(x_r)  # (B,L',D)
        H = self.fusion(h_n, h_r)        # (B,L',D)
        H = self.pool(H, x_r)            # (B,D)
        return self.head(H)              # (B,M)