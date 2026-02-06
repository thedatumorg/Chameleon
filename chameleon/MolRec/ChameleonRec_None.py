import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------------------
# Minimal ModernTCN components (unchanged)
# ------------------------------------------------------------------------
class Flatten_Head(nn.Module):
    def __init__(self, individual, n_vars, nf, target_window, head_dropout=0.0):
        super().__init__()
        self.individual = individual
        self.n_vars = n_vars
        if self.individual:
            self.linears  = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for _ in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(nf, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear  = nn.Linear(nf, target_window)
            self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:, i, :, :])
                z = self.linears[i](z)
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)
        else:
            x = self.flatten(x)
            x = self.linear(x)
            x = self.dropout(x)
        return x

def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups, dilation=1, bias=False):
    if padding is None:
        padding = kernel_size // 2
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride,
                  padding=padding, dilation=dilation, groups=groups, bias=bias),
        nn.BatchNorm1d(out_channels),
    )

def fuse_bn(conv, bn):
    kernel = conv.weight
    running_mean = bn.running_mean
    running_var  = bn.running_var
    gamma = bn.weight
    beta  = bn.bias
    eps   = bn.eps
    std = (running_var + eps).sqrt()
    t = (gamma / std).reshape(-1, 1, 1)
    return kernel * t, beta - running_mean * gamma / std

class ReparamLargeKernelConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, groups, small_kernel, small_kernel_merged=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.small_kernel = small_kernel
        padding = kernel_size // 2
        if small_kernel_merged:
            self.lkb_reparam = nn.Conv1d(in_channels, out_channels, kernel_size,
                                         stride=stride, padding=padding, dilation=1,
                                         groups=groups, bias=True)
        else:
            self.lkb_origin = conv_bn(in_channels, out_channels, kernel_size,
                                      stride=stride, padding=padding, dilation=1, groups=groups, bias=False)
            if small_kernel is not None:
                assert small_kernel <= kernel_size, 'small_kernel cannot exceed kernel_size'
                self.small_conv = conv_bn(in_channels, out_channels, small_kernel,
                                          stride=stride, padding=small_kernel // 2, groups=groups, dilation=1, bias=False)

    def forward(self, x):
        if hasattr(self, 'lkb_reparam'):
            return self.lkb_reparam(x)
        out = self.lkb_origin(x)
        if hasattr(self, 'small_conv'):
            out = out + self.small_conv(x)
        return out

    def _pad_kernel_sides(self, k, left, right, pad_value=0.0):
        D_out, D_in, ks = k.shape
        if pad_value == 0:
            pad_left  = torch.zeros(D_out, D_in, left,  device=k.device, dtype=k.dtype)
            pad_right = torch.zeros(D_out, D_in, right, device=k.device, dtype=k.dtype)
        else:
            pad_left  = torch.ones(D_out, D_in, left,  device=k.device, dtype=k.dtype) * pad_value
            pad_right = torch.ones(D_out, D_in, right, device=k.device, dtype=k.dtype) * pad_value
        return torch.cat([pad_left, k, pad_right], dim=-1)

    def get_equivalent_kernel_bias(self):
        eq_k, eq_b = fuse_bn(self.lkb_origin[0], self.lkb_origin[1])
        if hasattr(self, 'small_conv'):
            small_k, small_b = fuse_bn(self.small_conv[0], self.small_conv[1])
            eq_b = eq_b + small_b
            pad = (self.kernel_size - self.small_kernel) // 2
            small_k_padded = self._pad_kernel_sides(small_k, pad, pad, 0.0)
            eq_k = eq_k + small_k_padded
        return eq_k, eq_b

    def merge_kernel(self):
        if hasattr(self, 'lkb_reparam'):
            return
        eq_k, eq_b = self.get_equivalent_kernel_bias()
        conv0 = self.lkb_origin[0]
        self.lkb_reparam = nn.Conv1d(in_channels=conv0.in_channels,
                                     out_channels=conv0.out_channels,
                                     kernel_size=conv0.kernel_size, stride=conv0.stride,
                                     padding=conv0.padding, dilation=conv0.dilation,
                                     groups=conv0.groups, bias=True)
        self.lkb_reparam.weight.data = eq_k
        self.lkb_reparam.bias.data   = eq_b
        del self.lkb_origin
        if hasattr(self, 'small_conv'):
            del self.small_conv

class _Block(nn.Module):
    def __init__(self, large_size, small_size, dmodel, dff, nvars, small_kernel_merged=False, drop=0.1):
        super().__init__()
        self.dw = ReparamLargeKernelConv(in_channels=nvars * dmodel, out_channels=nvars * dmodel,
                                         kernel_size=large_size, stride=1, groups=nvars * dmodel,
                                         small_kernel=small_size, small_kernel_merged=small_kernel_merged)
        self.norm = nn.BatchNorm1d(dmodel)

        self.ffn1pw1 = nn.Conv1d(in_channels=nvars * dmodel, out_channels=nvars * dff, kernel_size=1, stride=1,
                                 padding=0, dilation=1, groups=nvars)
        self.ffn1act = nn.GELU()
        self.ffn1pw2 = nn.Conv1d(in_channels=nvars * dff, out_channels=nvars * dmodel, kernel_size=1, stride=1,
                                 padding=0, dilation=1, groups=nvars)
        self.ffn1drop1 = nn.Dropout(drop)
        self.ffn1drop2 = nn.Dropout(drop)

        self.ffn2pw1 = nn.Conv1d(in_channels=nvars * dmodel, out_channels=nvars * dff, kernel_size=1, stride=1,
                                 padding=0, dilation=1, groups=dmodel)
        self.ffn2act = nn.GELU()
        self.ffn2pw2 = nn.Conv1d(in_channels=nvars * dff, out_channels=nvars * dmodel, kernel_size=1, stride=1,
                                 padding=0, dilation=1, groups=dmodel)
        self.ffn2drop1 = nn.Dropout(drop)
        self.ffn2drop2 = nn.Dropout(drop)

    def forward(self, x):
        inp = x
        B, M, D, N = x.shape
        x = x.reshape(B, M * D, N)
        x = self.dw(x)
        x = x.reshape(B * M, D, N)
        x = self.norm(x)
        x = x.reshape(B, M, D, N)

        x = x.reshape(B, M * D, N)
        x = self.ffn1drop1(self.ffn1pw1(x))
        x = self.ffn1act(x)
        x = self.ffn1drop2(self.ffn1pw2(x))
        x = x.reshape(B, M, D, N)

        x = x.permute(0, 2, 1, 3)
        x = x.reshape(B, D * M, N)
        x = self.ffn2drop1(self.ffn2pw1(x))
        x = self.ffn2act(x)
        x = self.ffn2drop2(self.ffn2pw2(x))
        x = x.reshape(B, D, M, N)
        x = x.permute(0, 2, 1, 3)
        return inp + x

class _Stage(nn.Module):
    def __init__(self, ffn_ratio, num_blocks, large_size, small_size, dmodel, nvars,
                 small_kernel_merged=False, drop=0.1):
        super().__init__()
        d_ffn = dmodel * ffn_ratio
        self.blocks = nn.ModuleList([
            _Block(large_size=large_size, small_size=small_size,
                   dmodel=dmodel, dff=d_ffn, nvars=nvars,
                   small_kernel_merged=small_kernel_merged, drop=drop)
            for _ in range(num_blocks)
        ])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x

class ModernTCN(nn.Module):
    def __init__(self, task_name, patch_size, patch_stride, stem_ratio, downsample_ratio,
                 ffn_ratio, num_blocks, large_size, small_size, dims, dw_dims,
                 nvars, small_kernel_merged=False, backbone_dropout=0.1, head_dropout=0.1,
                 use_multi_scale=True, revin=False, affine=True, subtract_last=False,
                 freq=None, seq_len=512, c_in=7, individual=False, target_window=96,
                 class_drop=0., class_num=10):
        super().__init__()

        self.task_name = task_name
        self.revin = revin
        if self.revin:
            print("Warning: RevIN requested but not implemented; proceeding without RevIN.")
            self.revin = False

        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv1d(1, dims[0], kernel_size=patch_size, stride=patch_stride),
            nn.BatchNorm1d(dims[0])
        )
        self.downsample_layers.append(stem)

        self.num_stage = len(num_blocks)
        if self.num_stage > 1:
            for i in range(self.num_stage - 1):
                downsample_layer = nn.Sequential(
                    nn.BatchNorm1d(dims[i]),
                    nn.Conv1d(dims[i], dims[i + 1], kernel_size=downsample_ratio, stride=downsample_ratio),
                )
                self.downsample_layers.append(downsample_layer)

        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.downsample_ratio = downsample_ratio

        self.stages = nn.ModuleList()
        for stage_idx in range(self.num_stage):
            layer = _Stage(ffn_ratio, num_blocks[stage_idx],
                           large_size[stage_idx], small_size[stage_idx],
                           dmodel=dims[stage_idx], nvars=nvars,
                           small_kernel_merged=small_kernel_merged, drop=backbone_dropout)
            self.stages.append(layer)

        patch_num = seq_len // patch_stride
        self.n_vars = c_in
        self.individual = individual
        self.d_model_last = dims[self.num_stage - 1]
        self.use_multi_scale = use_multi_scale
        self.class_drop = class_drop
        self.class_num  = class_num
        if use_multi_scale:
            self.head_nf = self.d_model_last * patch_num
            self.head = Flatten_Head(self.individual, self.n_vars, self.head_nf, target_window, head_dropout=head_dropout)
        else:
            if patch_num % (downsample_ratio ** (self.num_stage - 1)) == 0:
                self.head_nf = self.d_model_last * patch_num // (downsample_ratio ** (self.num_stage - 1))
            else:
                self.head_nf = self.d_model_last * (patch_num // (downsample_ratio ** (self.num_stage - 1)) + 1)
            self.head = Flatten_Head(self.individual, self.n_vars, self.head_nf, target_window, head_dropout=head_dropout)

        self.act_class = F.gelu
        self.class_dropout = nn.Dropout(self.class_drop)
        self.head_class = nn.Linear(self.n_vars * self.head_nf, self.class_num)

    def forward_feature(self, x, te=None):
        B, M, L = x.shape
        x = x.unsqueeze(-2)  # (B,M,1,L)

        for i in range(self.num_stage):
            B, M, D, N = x.shape
            z = x.reshape(B * M, D, N)
            if i == 0:
                if self.patch_size != self.patch_stride:
                    pad_len = self.patch_size - self.patch_stride
                    pad = z[:, :, -1:].repeat(1, 1, pad_len)
                    z = torch.cat([z, pad], dim=-1)
            else:
                if N % self.downsample_ratio != 0:
                    pad_len = self.downsample_ratio - (N % self.downsample_ratio)
                    z = torch.cat([z, z[:, :, -pad_len:]], dim=-1)
            z = self.downsample_layers[i](z)
            _, D_, N_ = z.shape
            x = z.reshape(B, M, D_, N_)
            x = self.stages[i](x)
        return x

    def structural_reparam(self):
        for m in self.modules():
            if hasattr(m, 'merge_kernel'):
                m.merge_kernel()

# ------------------------------------------------------------------------
# ModernTCN Encoder wrapper: (B, C, L) -> (B, L_out, D_out)
# ------------------------------------------------------------------------
class ModernTCNEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        d_model: int,
        patch_size: int = 8,
        patch_stride: int = 8,
        downsample_ratio: int = 2,
        num_blocks = (2, 2),
        large_size  = (15, 15),
        small_size  = (3, 3),
        dims        = (128, 256),
        dw_dims     = (128, 256),
        backbone_dropout: float = 0.1,
        use_multi_scale: bool = True,
        revin: bool = False,
    ):
        super().__init__()
        self.backbone = ModernTCN(
            task_name='classification',
            patch_size=patch_size,
            patch_stride=patch_stride,
            stem_ratio=None,
            downsample_ratio=downsample_ratio,
            ffn_ratio=4,
            num_blocks=num_blocks,
            large_size=large_size,
            small_size=small_size,
            dims=dims,
            dw_dims=dw_dims,
            nvars=in_channels,
            small_kernel_merged=False,
            backbone_dropout=backbone_dropout,
            head_dropout=0.0,
            use_multi_scale=use_multi_scale,
            revin=revin,
            affine=True,
            subtract_last=False,
            seq_len=512,
            c_in=in_channels,
            individual=False,
            target_window=1,
            class_num=10
        )

        self.final_dim = int(dims[-1])
        self.out_dim   = int(d_model)
        self.var_fusion = 'mean'
        if self.final_dim != self.out_dim or self.var_fusion == 'concat':
            in_linear = self.final_dim if self.var_fusion == 'mean' else in_channels * self.final_dim
            self.proj = nn.Sequential(
                nn.LayerNorm(in_linear),
                nn.Linear(in_linear, self.out_dim),
            )
        else:
            self.proj = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, L = x.shape
        feats = self.backbone.forward_feature(x)   # (B, M, Df, N)
        B, M, Df, N = feats.shape

        if self.var_fusion == 'mean':
            fused = feats.mean(dim=1)              # (B, Df, N)
            fused = fused.transpose(1, 2)          # (B, N, Df)
        else:
            fused = feats.permute(0, 3, 1, 2).contiguous().view(B, N, M * Df)

        out = self.proj(fused)                     # (B, N, D_out)
        return out

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
        resid_energy = (resid**2).mean(dim=1)                                  # (B,L)
        resid_pooled = F.adaptive_avg_pool1d(resid_energy.unsqueeze(1), Lp).squeeze(1)  # (B,L')
        resid_prior  = (resid_pooled - resid_pooled.mean(dim=1, keepdim=True)) / \
                       (resid_pooled.std(dim=1, keepdim=True) + 1e-6)

        s = self.score(feats).squeeze(-1)  # (B,L')
        logits = s + resid_prior
        w = torch.softmax(logits, dim=1).unsqueeze(-1)  # (B,L',1)
        return (feats * w).sum(dim=1)                   # (B,D)


class ChameleonRec_None(nn.Module):
    def __init__(
        self, in_channels, num_models,
        d_model=256,
        patch_size=8, patch_stride=8, downsample_ratio=2,
        num_blocks=(2, 2), large_size=(15, 15), small_size=(3, 3),
        dims=(128, 256), dw_dims=(128, 256),
        dropout=0.1
    ):
        super().__init__()

        self.d_model = d_model

        self.encoder_normal = ModernTCNEncoder(
            in_channels=in_channels, d_model=d_model,
            patch_size=patch_size, patch_stride=patch_stride, downsample_ratio=downsample_ratio,
            num_blocks=num_blocks, large_size=large_size, small_size=small_size,
            dims=dims, dw_dims=dw_dims, backbone_dropout=dropout, revin=False
        )

        self.pool   = ResidualAttentionPool(d_model)
        self.head   = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, num_models))

    def forward(self, x_n):  # both (B,C,L)
        h_n = self.encoder_normal(x_n)     # (B,L',D)
        H   = self.pool(h_n, x_n)            # (B,D)
        return self.head(H)                # (B,M)