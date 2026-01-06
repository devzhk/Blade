import numpy as np
import torch

from utils import persistence
import torch.nn.functional as F


#----------------------------------------------------------------------------
# Unified routine for initializing weights and biases.

def weight_init(shape, mode, fan_in, fan_out):
    if mode == 'xavier_uniform': return np.sqrt(6 / (fan_in + fan_out)) * (torch.rand(*shape) * 2 - 1)
    if mode == 'xavier_normal':  return np.sqrt(2 / (fan_in + fan_out)) * torch.randn(*shape)
    if mode == 'kaiming_uniform': return np.sqrt(3 / fan_in) * (torch.rand(*shape) * 2 - 1)
    if mode == 'kaiming_normal':  return np.sqrt(1 / fan_in) * torch.randn(*shape)
    raise ValueError(f'Invalid init mode "{mode}"')

#----------------------------------------------------------------------------
# Fully-connected layer.

@persistence.persistent_class
class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, init_mode='kaiming_normal', init_weight=1, init_bias=0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        init_kwargs = dict(mode=init_mode, fan_in=in_features, fan_out=out_features)
        self.weight = torch.nn.Parameter(weight_init([out_features, in_features], **init_kwargs) * init_weight)
        self.bias = torch.nn.Parameter(weight_init([out_features], **init_kwargs) * init_bias) if bias else None

    def forward(self, x):
        x = x @ self.weight.to(x.dtype).t()
        if self.bias is not None:
            x = x.add_(self.bias.to(x.dtype))
        return x

#----------------------------------------------------------------------------
# Convolutional layer with optional up/downsampling.

@persistence.persistent_class
class Conv2d(torch.nn.Module):
    def __init__(self,
        in_channels, out_channels, kernel, bias=True, up=False, down=False,
        resample_filter=[1,1], fused_resample=False, init_mode='kaiming_normal', init_weight=1, init_bias=0,
    ):
        assert not (up and down)
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up = up
        self.down = down
        self.fused_resample = fused_resample
        init_kwargs = dict(mode=init_mode, fan_in=in_channels*kernel*kernel, fan_out=out_channels*kernel*kernel)
        self.weight = torch.nn.Parameter(weight_init([out_channels, in_channels, kernel, kernel], **init_kwargs) * init_weight) if kernel else None
        self.bias = torch.nn.Parameter(weight_init([out_channels], **init_kwargs) * init_bias) if kernel and bias else None
        f = torch.as_tensor(resample_filter, dtype=torch.float32)
        f = f.ger(f).unsqueeze(0).unsqueeze(1) / f.sum().square()
        self.register_buffer('resample_filter', f if up or down else None)

    def forward(self, x):
        w = self.weight.to(x.dtype) if self.weight is not None else None
        b = self.bias.to(x.dtype) if self.bias is not None else None
        f = self.resample_filter.to(x.dtype) if self.resample_filter is not None else None
        w_pad = w.shape[-1] // 2 if w is not None else 0
        f_pad = (f.shape[-1] - 1) // 2 if f is not None else 0

        if self.fused_resample and self.up and w is not None:
            x = torch.nn.functional.conv_transpose2d(x, f.mul(4).tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=max(f_pad - w_pad, 0))
            x = torch.nn.functional.conv2d(x, w, padding=max(w_pad - f_pad, 0))
        elif self.fused_resample and self.down and w is not None:
            x = torch.nn.functional.conv2d(x, w, padding=w_pad+f_pad)
            x = torch.nn.functional.conv2d(x, f.tile([self.out_channels, 1, 1, 1]), groups=self.out_channels, stride=2)
        else:
            if self.up:
                x = torch.nn.functional.conv_transpose2d(x, f.mul(4).tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=f_pad)
            if self.down:
                x = torch.nn.functional.conv2d(x, f.tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=f_pad)
            if w is not None:
                x = torch.nn.functional.conv2d(x, w, padding=w_pad)
        if b is not None:
            x = x.add_(b.reshape(1, -1, 1, 1))
        return x

#----------------------------------------------------------------------------
# Group normalization.

@persistence.persistent_class
class GroupNorm(torch.nn.Module):
    def __init__(self, num_channels, num_groups=32, min_channels_per_group=4, eps=1e-5):
        super().__init__()
        self.num_groups = min(num_groups, num_channels // min_channels_per_group)
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(num_channels))
        self.bias = torch.nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        x = torch.nn.functional.group_norm(x, num_groups=self.num_groups, weight=self.weight.to(x.dtype), bias=self.bias.to(x.dtype), eps=self.eps)
        return x


#----------------------------------------------------------------------------
# Unified U-Net block with optional up/downsampling and self-attention.
# Represents the union of all features employed by the DDPM++, NCSN++, and
# ADM architectures.

@persistence.persistent_class
class UNetBlock(torch.nn.Module):
    def __init__(self,
        in_channels, out_channels, emb_channels, up=False, down=False, attention=False,
        num_heads=None, channels_per_head=64, dropout=0, skip_scale=1, eps=1e-5,
        resample_filter=[1,1], resample_proj=False, adaptive_scale=True,
        init=dict(), init_zero=dict(init_weight=0), init_attn=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.emb_channels = emb_channels
        self.num_heads = 0 if not attention else num_heads if num_heads is not None else out_channels // channels_per_head
        self.dropout = dropout
        self.skip_scale = skip_scale
        self.adaptive_scale = adaptive_scale

        self.norm0 = GroupNorm(num_channels=in_channels, eps=eps)
        self.conv0 = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel=3, up=up, down=down, resample_filter=resample_filter, **init)
        self.affine = Linear(in_features=emb_channels, out_features=out_channels*(2 if adaptive_scale else 1), **init)
        self.norm1 = GroupNorm(num_channels=out_channels, eps=eps)
        self.conv1 = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=3, **init_zero)

        self.skip = None
        if out_channels != in_channels or up or down:
            kernel = 1 if resample_proj or out_channels!= in_channels else 0
            self.skip = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel=kernel, up=up, down=down, resample_filter=resample_filter, **init)

        if self.num_heads:
            self.norm2 = GroupNorm(num_channels=out_channels, eps=eps)
            self.qkv = Conv2d(in_channels=out_channels, out_channels=out_channels*3, kernel=1, **(init_attn if init_attn is not None else init))
            self.proj = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=1, **init_zero)

    def forward(self, x, emb):
        orig = x
        x = self.conv0(F.silu(self.norm0(x)))

        params = self.affine(emb).unsqueeze(2).unsqueeze(3).to(x.dtype)
        if self.adaptive_scale:
            scale, shift = params.chunk(chunks=2, dim=1)
            x = F.silu(torch.addcmul(shift, self.norm1(x), scale + 1))
        else:
            x = F.silu(self.norm1(x.add_(params)))

        x = self.conv1(torch.nn.functional.dropout(x, p=self.dropout, training=self.training))
        x = x.add_(self.skip(orig) if self.skip is not None else orig)
        x = x * self.skip_scale

        if self.num_heads:
            q, k, v = self.qkv(self.norm2(x)).reshape(x.shape[0] * self.num_heads, x.shape[1] // self.num_heads, 3, -1).permute(0, 3, 2, 1).unbind(2)
            # q shape: (B * num_heads, H * W, C // num_heads)
            a = F.scaled_dot_product_attention(q, k, v).permute(0, 2, 1) # (B * num_heads, C // num_heads, H * W)

            x = self.proj(a.reshape(*x.shape)).add_(x)
            x = x * self.skip_scale
        return x



#----------------------------------------------------------------------------
# Timestep embedding used in the DDPM++ and ADM architectures.

@persistence.persistent_class
class PositionalEmbedding(torch.nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(start=0, end=self.num_channels//2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x

#----------------------------------------------------------------------------
# Timestep embedding used in the NCSN++ architecture.

@persistence.persistent_class
class FourierEmbedding(torch.nn.Module):
    def __init__(self, num_channels, scale=16):
        super().__init__()
        self.register_buffer('freqs', torch.randn(num_channels // 2) * scale)

    def forward(self, x):
        x = x.ger((2 * np.pi * self.freqs).to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x
    
#----------------------------------------------------------------------------
# Conditional U-Net block with cross-attention instead of self-attention.

@persistence.persistent_class
class ConditionalUNetBlock(torch.nn.Module):
    def __init__(
        self, in_channels, out_channels, emb_channels, context_dim,
        up=False, down=False, use_cross_attention=False,
        num_heads=None, channels_per_head=64, dropout=0, skip_scale=1, eps=1e-5,
        resample_filter=[1,1], resample_proj=False, adaptive_scale=True,
        init=dict(), init_zero=dict(init_weight=0), init_attn=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.emb_channels = emb_channels
        self.use_cross_attention = use_cross_attention
        self.num_heads = 0
        self.dropout = dropout
        self.skip_scale = skip_scale
        self.adaptive_scale = adaptive_scale

        # same conv+norm layers as UNetBlock
        self.norm0 = GroupNorm(num_channels=in_channels, eps=eps)
        self.conv0 = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel=3, up=up, down=down, resample_filter=resample_filter, **init)
        self.affine = Linear(in_features=emb_channels, out_features=out_channels*(2 if adaptive_scale else 1), **init)
        self.norm1 = GroupNorm(num_channels=out_channels, eps=eps)
        self.conv1 = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=3, **init_zero)

        self.skip = None
        if out_channels != in_channels or up or down:
            kernel = 1 if resample_proj or out_channels!= in_channels else 0
            self.skip = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel=kernel, up=up, down=down, resample_filter=resample_filter, **init)

        # cross-attention module
        if use_cross_attention:
            head_dim = channels_per_head
            num_heads = num_heads or (out_channels // head_dim)
            self.cross_attn = CrossAttentionBlock(
                in_channels=out_channels,
                context_dim=context_dim,
                num_heads=num_heads,
                head_dim=head_dim
            )

    def forward(self, x, emb, context=None):
        orig = x
        x = self.conv0(F.silu(self.norm0(x)))

        params = self.affine(emb).unsqueeze(2).unsqueeze(3).to(x.dtype)
        if self.adaptive_scale:
            scale, shift = params.chunk(chunks=2, dim=1)
            x = F.silu(torch.addcmul(shift, self.norm1(x), scale + 1))
        else:
            x = F.silu(self.norm1(x.add_(params)))

        x = self.conv1(torch.nn.functional.dropout(x, p=self.dropout, training=self.training))
        x = x.add_(self.skip(orig) if self.skip is not None else orig)
        x = x * self.skip_scale

        # cross-attend instead of self-attn
        if self.use_cross_attention and context is not None:
            x = self.cross_attn(x, context)

        return x
    
@persistence.persistent_class
class CrossAttentionBlock(torch.nn.Module):
    def __init__(self, in_channels, context_dim, num_heads=None, head_dim=64):
        super().__init__()
        self.context_proj_K = Linear(context_dim, in_channels, bias=False)
        self.context_proj_V = Linear(context_dim, in_channels, bias=False)

        self.attn = torch.nn.MultiheadAttention(embed_dim=in_channels,
                                                num_heads=num_heads or in_channels//head_dim,
                                                batch_first=True)

        self.proj_out = Conv2d(in_channels, in_channels, kernel=1,
                               init_mode='kaiming_uniform', init_weight=0, init_bias=0)
    def forward(self, x, context):
        B, C, H, W = x.shape

        x_flat = x.view(B, C, H*W).permute(0, 2, 1)  # [B, L, C]

        mem_K = self.context_proj_K(context).unsqueeze(1)
        mem_V = self.context_proj_V(context).unsqueeze(1)

        # cross‑attention: Q = x_flat, K = V = mem
        attn_out, _ = self.attn(x_flat, mem_K, mem_V)    # [B, L, C]

        # reshape back to [B,C,H,W] and run the 1×1 conv
        attn_out = attn_out.permute(0,2,1).view(B, C, H, W)
        attn_out = self.proj_out(attn_out)

        # residual add
        return x + attn_out
    
@persistence.persistent_class
class DirectUNetBlock(torch.nn.Module):
    def __init__(self,
        in_channels, out_channels,
        up=False, down=False, attention=False,
        num_heads=None, channels_per_head=64,
        dropout=0, skip_scale=1, eps=1e-5,
        resample_filter=[1,1], resample_proj=False,
        init=dict(), init_zero=dict(init_weight=0), init_attn=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.skip_scale = skip_scale

        # First conv
        self.norm0 = GroupNorm(num_channels=in_channels, eps=eps)
        self.conv0 = Conv2d(in_channels=in_channels, out_channels=out_channels,
                            kernel=3, up=up, down=down,
                            resample_filter=resample_filter, **init)

        # Second conv (no embedding conditioning)
        self.norm1 = GroupNorm(num_channels=out_channels, eps=eps)
        self.conv1 = Conv2d(in_channels=out_channels, out_channels=out_channels,
                            kernel=3, **init_zero)

        # Optional 1×1 skip / resample projection
        self.skip = None
        if out_channels != in_channels or up or down:
            kernel = 1 if (resample_proj or out_channels != in_channels) else 0
            self.skip = Conv2d(in_channels=in_channels, out_channels=out_channels,
                               kernel=kernel, up=up, down=down,
                               resample_filter=resample_filter, **init)

        # Optional self-attention
        self.num_heads = 0
        if attention:
            self.num_heads = num_heads if num_heads is not None else out_channels // channels_per_head
        if self.num_heads:
            self.norm2 = GroupNorm(num_channels=out_channels, eps=eps)
            attn_init = init_attn if init_attn is not None else init
            self.qkv  = Conv2d(in_channels=out_channels, out_channels=out_channels*3, kernel=1, **attn_init)
            self.proj = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=1, **init_zero)

    def forward(self, x):
        orig = x

        # ---- residual path ----
        x = self.conv0(F.silu(self.norm0(x)))

        x = F.silu(self.norm1(x))
        x = self.conv1(F.dropout(x, p=self.dropout, training=self.training))

        # add skip connection
        if self.skip is not None:
            x = x.add_(self.skip(orig))
        else:
            x = x.add_(orig)
        x = x * self.skip_scale

        # ---- optional attention ----
        if self.num_heads:
            # q/k/v: reshape [B, C→3C, H, W] → [B*nh, L, C//nh]
            b, c, h, w = x.shape
            nh = self.num_heads
            qkv = self.qkv(self.norm2(x))
            qkv = qkv.reshape(b, 3, nh, c//nh, h*w).permute(0,2,4,1,3)
            q, k, v = qkv.unbind(3)  # each is [B, nh, H*W, C//nh]
            q = q.reshape(b*nh, h*w, c//nh)
            k = k.reshape(b*nh, h*w, c//nh)
            v = v.reshape(b*nh, h*w, c//nh)

            a = F.scaled_dot_product_attention(q, k, v)
            a = a.reshape(b, nh, h*w, c//nh).permute(0,1,3,2)
            a = a.reshape(b, c, h, w)

            x = self.proj(a).add_(x)
            x = x * self.skip_scale

        return x
