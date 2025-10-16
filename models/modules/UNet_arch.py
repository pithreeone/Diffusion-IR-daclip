import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import functools

from .module_util import (
    SinusoidalPosEmb,
    RandomOrLearnedSinusoidalPosEmb,
    NonLinearity,
    Upsample, Downsample,
    default_conv,
    ResBlock, Upsampler,
    LinearAttention, Attention,
    PreNorm, Residual, LayerNorm,
    Identity)

from .attention import SpatialTransformer


class UNet(nn.Module):
    # def __init__(self, in_nc, out_nc, nf, ch_mult=[1, 2, 4, 4], 
    #                 context_dim=512, use_degra_context=True, use_image_context=False, upscale=1):
    def __init__(self, in_nc, out_nc, nf, ch_mult=[1, 2, 4, 4], 
                    context_dim=512, use_daclip_context=True, upscale=1, in_ch_scale=2, attn=True):
        super().__init__()
        self.depth = len(ch_mult)
        self.upscale = upscale # not used
        self.context_dim = -1 if context_dim is None else context_dim
        self.use_image_context = self.use_degra_context = self.use_daclip_context = use_daclip_context
        self.attn = attn

        num_head_channels = 32
        dim_head = num_head_channels

        block_class = functools.partial(ResBlock, conv=default_conv, act=NonLinearity())

        # self.init_conv = default_conv(in_nc*2, nf, 7)
        self.init_conv = default_conv(in_nc*in_ch_scale, nf, 7)
        

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        ch_mult = [1] + ch_mult

        for i in range(self.depth):
            dim_in = nf * ch_mult[i]
            dim_out = nf * ch_mult[i+1]

            num_heads_in = dim_in // num_head_channels
            num_heads_out = dim_out // num_head_channels
            dim_head_in = dim_in // num_heads_in

            if self.attn:
                att_down = LinearAttention(dim_in) # if i < 2 else Attention(dim_in)
                att_up = LinearAttention(dim_out) # if i < 2 else Attention(dim_out)

            self.downs.append(nn.ModuleList([
                block_class(dim_in=dim_in, dim_out=dim_in),
                block_class(dim_in=dim_in, dim_out=dim_in),
                Residual(PreNorm(dim_in, att_down)) if attn else LayerNorm(dim_in),
                Downsample(dim_in, dim_out) if i != (self.depth-1) else default_conv(dim_in, dim_out)
            ]))

            self.ups.insert(0, nn.ModuleList([
                block_class(dim_in=dim_out + dim_in, dim_out=dim_out),
                block_class(dim_in=dim_out + dim_in, dim_out=dim_out),
                Residual(PreNorm(dim_in, att_down)) if attn else LayerNorm(dim_out),
                Upsample(dim_out, dim_in) if i!=0 else default_conv(dim_out, dim_in)
            ]))

        mid_dim = nf * ch_mult[-1]
        num_heads_mid = mid_dim // num_head_channels
        self.mid_block1 = block_class(dim_in=mid_dim, dim_out=mid_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim))) if attn else LayerNorm(mid_dim)
        self.mid_block2 = block_class(dim_in=mid_dim, dim_out=mid_dim)

        self.final_res_block = block_class(dim_in=nf * 2, dim_out=nf)
        self.final_conv = nn.Conv2d(nf, out_nc, 3, 1, 1)

    def get_last_layer(self):
        return self.final_conv.weight

    def check_image_size(self, x, h, w):
        s = int(math.pow(2, self.depth))
        mod_pad_h = (s - h % s) % s
        mod_pad_w = (s - w % s) % s
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward(self, input, cond=None, text_context=None, image_context=None):
        if cond is not None:
            x = torch.cat([input, cond], dim=1)
        else:
            x = input

        H, W = x.shape[2:]
        x = self.check_image_size(x, H, W)

        x = self.init_conv(x)
        x_ = x.clone()

        h = []
        for b1, b2, attn, downsample in self.downs:
            x = b1(x)
            h.append(x)

            x = b2(x)
            x = attn(x, context=image_context) if self.attn else attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x)
        x = self.mid_attn(x, context=image_context) if self.attn else self.mid_attn(x)
        x = self.mid_block2(x)

        for b1, b2, attn, upsample in self.ups:
            x = torch.cat([x, h.pop()], dim=1)
            x = b1(x)
            
            x = torch.cat([x, h.pop()], dim=1)
            x = b2(x)

            x = attn(x, context=image_context) if self.attn else attn(x)
            x = upsample(x)

        x = torch.cat([x, x_], dim=1)

        x = self.final_res_block(x)
        x = self.final_conv(x)

        x = x[..., :H, :W].contiguous()
        
        return x



