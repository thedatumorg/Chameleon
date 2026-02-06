import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ------------------------------------------------------------------------
# ConvNet backbone (from CNN baseline)
# ------------------------------------------------------------------------
class ConvNet(nn.Module):
    def __init__(
        self,
        original_length,
        num_blocks=5,
        kernel_size=3,
        padding=1,
        original_dim=1,
        num_classes=12
    ):
        super(ConvNet, self).__init__()

        self.num_class = num_classes
        self.kernel_size = kernel_size
        self.padding = padding
        self.layers = []

        dims = [original_dim]
        dims += list(2 ** np.arange(6, 6 + num_blocks))
        dims = [x if x <= 256 else 256 for x in dims]

        for i in range(num_blocks):
            self.layers.extend([
                nn.Conv1d(dims[i], dims[i + 1], kernel_size=self.kernel_size, padding=self.padding),
                nn.BatchNorm1d(dims[i + 1]),
                nn.ReLU(),
            ])
        self.layers.extend([
            nn.Conv1d(dims[-1], dims[-1], kernel_size=self.kernel_size, padding=self.padding),
            nn.ReLU(),
        ])
        self.layers = nn.Sequential(*self.layers)

        self.GAP = nn.AvgPool1d(original_length)

        self.fc1 = nn.Sequential(
            nn.Linear(dims[-1], num_classes)
        )

    def forward(self, x):
        # x: (B, C, L) with C=original_dim, L=original_length
        out = self.layers(x)
        out = self.GAP(out)                 # (B, dims[-1], 1)
        out = out.reshape(out.size(0), -1)  # (B, dims[-1])
        out = self.fc1(out)                 # (B, num_classes)
        return out


# ------------------------------------------------------------------------
# Fusion of normal / residual global embeddings
# ------------------------------------------------------------------------
class GatedFusion(nn.Module):
    """
    Gating in global embedding space:
      h_n, h_a: (B, D) -> fused: (B, D)
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.LayerNorm(2 * d_model),
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, h_n, h_a):
        # h_n, h_a: (B, D)
        H = torch.cat([h_n, h_a], dim=-1)  # (B, 2D)
        g = self.gate(H)                   # (B, D)
        mixed = g * h_a + (1.0 - g) * h_n  # (B, D)
        return self.proj(mixed)            # (B, D)


# ------------------------------------------------------------------------
# Model with dual Conv-style backbones on normal/residual streams
# (takes x_n, x_r as inputs; decomposition is done in the dataset)
# ------------------------------------------------------------------------
class ChameleonRec_Conv(nn.Module):
    def __init__(
        self,
        in_channels,
        num_models,
        window_size: int,
        d_model: int = 256,
        conv_num_blocks: int = 5,
        conv_kernel_size: int = 3,
        conv_padding: int = 1
    ):
        super().__init__()

        # metadata
        self.d_model = d_model
        self.window_size = window_size

        # build dims: [C_in, 2^6, 2^7, ...] capped at 256
        dims = [in_channels]
        dims += list(2 ** np.arange(6, 6 + conv_num_blocks))
        dims = [int(x if x <= 256 else 256) for x in dims]

        # --- build conv stack factory so branches don't share weights ---
        def make_conv_stack():
            layers = []
            for i in range(conv_num_blocks):
                layers.extend([
                    nn.Conv1d(dims[i], dims[i + 1],
                              kernel_size=conv_kernel_size,
                              padding=conv_padding),
                    nn.BatchNorm1d(dims[i + 1]),
                    nn.ReLU(),
                ])
            layers.extend([
                nn.Conv1d(dims[-1], dims[-1],
                          kernel_size=conv_kernel_size,
                          padding=conv_padding),
                nn.ReLU(),
            ])
            return nn.Sequential(*layers)

        # Two independent branches
        self.encoder_normal = make_conv_stack()
        self.encoder_anomaly = make_conv_stack()

        # Global average pooling over time + linear projection to d_model
        self.gap_normal   = nn.AvgPool1d(window_size)
        self.gap_anomaly  = nn.AvgPool1d(window_size)
        self.proj_normal  = nn.Linear(dims[-1], d_model)
        self.proj_anomaly = nn.Linear(dims[-1], d_model)

        # fusion + head
        self.fusion = GatedFusion(d_model)  # expects (B, D)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_models),
        )

    def _encode_branch(self, x, conv, gap, proj):
        """
        x: (B, C, L)
        conv: conv stack
        gap:  AvgPool1d over L
        proj: Linear(dims[-1] -> d_model)
        """
        feats = conv(x)                   # (B, C_last, L)
        pooled = gap(feats).squeeze(-1)   # (B, C_last)
        h = proj(pooled)                  # (B, d_model)
        return h

    def forward(self, x_n, x_r):  # both (B, C, L)
        h_n = self._encode_branch(x_n, self.encoder_normal,
                                  self.gap_normal, self.proj_normal)   # (B, D)
        h_r = self._encode_branch(x_r, self.encoder_anomaly,
                                  self.gap_anomaly, self.proj_anomaly) # (B, D)

        H = self.fusion(h_n, h_r)        # (B, D)
        return self.head(H)              # (B, M)