import numpy as np
import torch
import torch.nn.functional as F
from utils import persistence
from models.modules import (Conv2d, GroupNorm, Linear, PositionalEmbedding,
                     ConditionalUNetBlock, DirectUNetBlock)

#----------------------------------------------------------------------------
@persistence.persistent_class
class ConditionalDhariwalUNet(torch.nn.Module):
    def __init__(
        self,
        img_resolution,
        in_channels,
        out_channels,
        # classifier-free guidance labels (optional)
        label_dim=0,
        augment_dim=0,

        # UNet base settings
        model_channels=192,
        channel_mult=[1,2,3,4],
        channel_mult_emb=4,
        num_blocks=3,
        attn_resolutions=[32,16,8],
        dropout=0.1,
        label_dropout=0,
    ):
        super().__init__()
        emb_channels = model_channels * channel_mult_emb
        init = dict(init_mode='kaiming_uniform', init_weight=np.sqrt(1/3), init_bias=np.sqrt(1/3))
        init_zero = dict(init_mode='kaiming_uniform', init_weight=0, init_bias=0)
        block_kwargs = dict(channels_per_head=64,
                            dropout=dropout,
                            init=init,
                            init_zero=init_zero)

        # noise + label mappings
        self.map_noise    = PositionalEmbedding(num_channels=model_channels)
        self.map_augment  = Linear(augment_dim, model_channels, bias=False, **init_zero) if augment_dim else None
        self.map_layer0   = Linear(model_channels, emb_channels, **init)
        self.map_layer1   = Linear(emb_channels,    emb_channels, **init)
        self.map_label    = Linear(label_dim,      emb_channels,
                                    bias=False,
                                    init_mode='kaiming_normal',
                                    init_weight=np.sqrt(label_dim)) if label_dim else None
        self.label_dropout = label_dropout
        
        # measurement encoder → emb_channels vector
        self.measure_encoder = torch.nn.Sequential(
            Conv2d(in_channels, model_channels, kernel=3, **init),
            GroupNorm(model_channels),
            torch.nn.SiLU(),
            Conv2d(model_channels, model_channels, kernel=4, down=True, **init),
            GroupNorm(model_channels),
            torch.nn.SiLU(),
            torch.nn.AdaptiveAvgPool2d(1),  # [B, C, 1,1]
            torch.nn.Flatten(),             # [B, C]
            Linear(model_channels, emb_channels, **init)
        )
        
        # build encoder with ConditionalUNetBlock
        self.enc = torch.nn.ModuleDict()
        cout = in_channels
        for level, mult in enumerate(channel_mult):
            res = img_resolution >> level
            if level == 0:
                cin = cout
                cout = model_channels * mult
                self.enc[f'{res}x{res}_conv'] = Conv2d(cin, cout, kernel=3, **init)
            else:
                # downsample + conditional block
                self.enc[f'{res}x{res}_down'] = ConditionalUNetBlock(
                    in_channels=cout,
                    out_channels=cout,
                    emb_channels=emb_channels,
                    context_dim=emb_channels,
                    down=True,
                    use_cross_attention=True,
                    **block_kwargs
                )
            for idx in range(num_blocks):
                cin = cout
                cout = model_channels * mult
                self.enc[f'{res}x{res}_block{idx}'] = ConditionalUNetBlock(
                    in_channels=cin,
                    out_channels=cout,
                    emb_channels=emb_channels,
                    context_dim=emb_channels,
                    use_cross_attention=(res in attn_resolutions),
                    **block_kwargs
                )
        skips = [b.out_channels for b in self.enc.values()]
        
        # build decoder similarly
        self.dec = torch.nn.ModuleDict()
        for level, mult in reversed(list(enumerate(channel_mult))):
            res = img_resolution >> level
            if level == len(channel_mult) - 1:
                # two initial blocks at bottleneck
                self.dec[f'{res}x{res}_in0'] = ConditionalUNetBlock(
                    in_channels=cout,
                    out_channels=cout,
                    emb_channels=emb_channels,
                    context_dim=emb_channels,
                    use_cross_attention=True,
                    **block_kwargs
                )
                self.dec[f'{res}x{res}_in1'] = ConditionalUNetBlock(
                    in_channels=cout,
                    out_channels=cout,
                    emb_channels=emb_channels,
                    context_dim=emb_channels,
                    use_cross_attention=True,
                    **block_kwargs
                )
            else:
                self.dec[f'{res}x{res}_up'] = ConditionalUNetBlock(
                    in_channels=cout,
                    out_channels=cout,
                    emb_channels=emb_channels,
                    context_dim=emb_channels,
                    up=True,
                    use_cross_attention=True,
                    **block_kwargs
                )
            for idx in range(num_blocks + 1):
                cin = cout + skips.pop()
                cout = model_channels * mult
                self.dec[f'{res}x{res}_block{idx}'] = ConditionalUNetBlock(
                    in_channels=cin,
                    out_channels=cout,
                    emb_channels=emb_channels,
                    context_dim=emb_channels,
                    use_cross_attention=(res in attn_resolutions),
                    **block_kwargs
                )

        # final normalization + output conv
        self.out_norm = GroupNorm(cout)
        self.out_conv = Conv2d(cout, out_channels, kernel=3, **init_zero)

    def forward(self, x, noise_labels, measurement, class_labels=None, augment_labels=None):
        # map measurement → context embedding
        B = x.shape[0]
        # measurement: either [B,C,H_m,W_m] or [C,H_m,W_m]
        if measurement.dim() == 3:
            measurement = measurement.unsqueeze(0).expand(B, -1, -1, -1)
        meas_emb = self.measure_encoder(measurement)

        # noise + label → emb
        emb = self.map_noise(noise_labels)
        if self.map_augment and augment_labels is not None:
            emb = emb + self.map_augment(augment_labels)
        emb = F.silu(self.map_layer0(emb))
        emb = self.map_layer1(emb)
        if self.map_label:
            tmp = class_labels
            if self.training and self.label_dropout:
                mask = (torch.rand([x.shape[0],1], device=x.device) >= self.label_dropout).to(tmp.dtype)
                tmp = tmp * mask
            emb = emb + self.map_label(tmp)
        emb = F.silu(emb)

        # encoder
        skips = []
        for block in self.enc.values():
            x = block(x, emb, context=meas_emb) if isinstance(block, ConditionalUNetBlock) else block(x)
            skips.append(x)

        # decoder
        for block in self.dec.values():
            if x.shape[1] != block.in_channels:
                x = torch.cat([x, skips.pop()], dim=1)
            x = block(x, emb, context=meas_emb) if isinstance(block, ConditionalUNetBlock) else block(x)

        x = self.out_conv(F.silu(self.out_norm(x)))
        return x

@persistence.persistent_class
class DirectUNet(torch.nn.Module):
    def __init__(
        self,
        img_resolution,                     # Image resolution at input/output.
        in_channels,                        # Number of color channels at input.
        out_channels,                       # Number of color channels at output.
        cond_channels       = 1,            # Number of color channels for condition.
        label_dim           = 0,            # Number of class labels, 0 = unconditional.
        augment_dim         = 0,            # Augmentation label dimensionality, 0 = no augmentation.

        model_channels      = 192,          # Base multiplier for the number of channels.
        channel_mult        = [1,2,3,4],    # Per-resolution multipliers for the number of channels.
        channel_mult_emb    = 4,            # Multiplier for the dimensionality of the embedding vector.
        num_blocks          = 3,            # Number of residual blocks per resolution.
        attn_resolutions    = [32,16,8],    # List of resolutions with self-attention.
        dropout             = 0.10,         # List of resolutions with self-attention.
        label_dropout       = 0,            # Dropout probability of class labels for classifier-free guidance.
    ):
        super().__init__()
        self.img_resolution = img_resolution

        # same inits as DhariwalUNet
        init      = dict(init_mode='kaiming_uniform', init_weight=np.sqrt(1/3), init_bias=np.sqrt(1/3))
        init_zero = dict(init_mode='kaiming_uniform', init_weight=0, init_bias=0)

        # kwargs for each DirectUNetBlock
        block_kwargs = dict(
            channels_per_head=64,
            dropout=dropout,
            init=init,
            init_zero=init_zero,
        )

        # ─── ENCODER ─────────────────────────────────────────
        self.enc = torch.nn.ModuleDict()
        cout = in_channels
        for level, mult in enumerate(channel_mult):
            res = img_resolution >> level
            # first op at this resolution is either plain conv (level=0) or strided-down block
            if level == 0:
                cin  = cout
                cout = model_channels * mult
                self.enc[f'{res}x{res}_conv'] = Conv2d(
                    in_channels=cin, out_channels=cout, kernel=3, **init
                )
            else:
                # downsample block
                self.enc[f'{res}x{res}_down'] = DirectUNetBlock(
                    in_channels=cout,
                    out_channels=cout,
                    down=True,
                    attention=False,
                    **block_kwargs
                )
            # then several residual blocks
            for idx in range(num_blocks):
                cin  = cout
                cout = model_channels * mult
                self.enc[f'{res}x{res}_block{idx}'] = DirectUNetBlock(
                    in_channels=cin,
                    out_channels=cout,
                    attention=(res in attn_resolutions),
                    **block_kwargs
                )

        # record skip‐connection channels
        skips = [blk.out_channels for blk in self.enc.values()]

        # ─── DECODER ─────────────────────────────────────────
        self.dec = torch.nn.ModuleDict()
        for level, mult in reversed(list(enumerate(channel_mult))):
            res = img_resolution >> level
            # first block at coarsest level: two plain blocks
            if level == len(channel_mult) - 1:
                self.dec[f'{res}x{res}_in0'] = DirectUNetBlock(
                    in_channels=cout, out_channels=cout,
                    attention=True,
                    **block_kwargs
                )
                self.dec[f'{res}x{res}_in1'] = DirectUNetBlock(
                    in_channels=cout, out_channels=cout,
                    attention=False,
                    **block_kwargs
                )
            else:
                # upsample block
                self.dec[f'{res}x{res}_up'] = DirectUNetBlock(
                    in_channels=cout,
                    out_channels=cout,
                    up=True,
                    attention=False,
                    **block_kwargs
                )
            # then residual blocks that take a skip‐connection concat
            for idx in range(num_blocks + 1):
                cin  = cout + skips.pop()
                cout = model_channels * mult
                self.dec[f'{res}x{res}_block{idx}'] = DirectUNetBlock(
                    in_channels=cin,
                    out_channels=cout,
                    attention=(res in attn_resolutions),
                    **block_kwargs
                )

        # final conv to bring back to out_channels
        self.out_norm = GroupNorm(num_channels=cout)
        self.out_conv = Conv2d(
            in_channels=cout, out_channels=out_channels,
            kernel=3,
            **init_zero
        )

    def forward(self, x):
        
        x = F.interpolate(
                x,
                scale_factor=2,
                mode='nearest'
        )
        # Encoder
        skips = []
        for key, block in self.enc.items():
            x = block(x) if isinstance(block, DirectUNetBlock) else block(x)
            skips.append(x)

        # Decoder
        for key, block in self.dec.items():
            # when it's a concat‐block, we need to pull from skips
            if x.shape[1] != block.in_channels:
                x = torch.cat([x, skips.pop()], dim=1)
            x = block(x)

        # Final conv
        x = self.out_conv(F.silu(self.out_norm(x)))
        return x