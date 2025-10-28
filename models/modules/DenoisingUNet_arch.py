import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
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
from .diffusionmodules.openaimodel import AttentionBlock

class SpatialModulationBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels=None):
        super().__init__()
        hidden_channels = hidden_channels or in_channels // 2
        
        # 1x1 convs to generate modulation map
        self.modulator = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, 1),  # output spatial weight map
            nn.Sigmoid()  # ensure weights in [0,1]
        )

    def forward(self, cond_feat):
        """
        cond_feat: condition feature map (B, C, H, W)
        dec_feat: decoder feature map (B, C, H, W)
        """
        w = self.modulator(cond_feat)  # (B, 1, H, W)
        out = cond_feat * w  # modulate and keep residual
        return out, w

class ConditionalUNet(nn.Module):
    # def __init__(self, in_nc, out_nc, nf, ch_mult=[1, 2, 4, 4], 
    #                 context_dim=512, use_degra_context=True, use_image_context=False, upscale=1):
    def __init__(self, in_nc, out_nc, nf, ch_mult=[1, 2, 4, 4], 
                    context_dim=512, use_daclip_context=True, upscale=1, in_ch_scale=2, cond_type=None, attn=True, use_deg_embedding=False):
        super().__init__()
        self.depth = len(ch_mult)
        self.upscale = upscale # not used
        # self.context_dim = -1 if context_dim is None else context_dim
        # self.use_image_context = use_image_context
        # self.use_degra_context = use_degra_context
        self.use_image_context = self.use_degra_context = self.use_daclip_context = use_daclip_context
        self.cond_type = cond_type
        self.use_deg_embedding = use_deg_embedding
        self.attn = attn

        num_head_channels = 32
        dim_head = num_head_channels

        block_class = functools.partial(ResBlock, conv=default_conv, act=NonLinearity())

        # self.init_conv = default_conv(in_nc*2, nf, 7)
        self.init_conv = default_conv(in_nc*in_ch_scale, nf, 7)
        if self.cond_type == 'cond_module':
            self.init_conv_cond = default_conv(in_nc, nf, 7)
            self.init_conv_cond = default_conv(in_nc*2, nf, 7)
        if self.use_deg_embedding:
            self.init_conv_res_emb = default_conv(in_nc, nf, 7)
        # time embeddings
        time_dim = nf * 4

        self.random_or_learned_sinusoidal_cond = False

        if self.random_or_learned_sinusoidal_cond:
            learned_sinusoidal_dim = 16
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, False)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(nf)
            fourier_dim = nf

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # if self.context_dim > 0 and self.use_degra_context: 
        #     self.prompt = nn.Parameter(torch.rand(1, time_dim))
        #     self.text_mlp = nn.Sequential(
        #         nn.Linear(context_dim, time_dim), NonLinearity(),
        #         nn.Linear(time_dim, time_dim))
        #     self.prompt_mlp = nn.Linear(time_dim, time_dim)

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        self.cond_downs = nn.ModuleList([])
        self.residual_emb_downs = nn.ModuleList([])
        ch_mult = [1] + ch_mult

        self.SMblocks = nn.ModuleList([])

        for i in range(self.depth):
            dim_in = nf * ch_mult[i]
            dim_out = nf * ch_mult[i+1]

            num_heads_in = dim_in // num_head_channels
            num_heads_out = dim_out // num_head_channels
            dim_head_in = dim_in // num_heads_in

            # if self.use_image_context and context_dim > 0:
            #     att_down = LinearAttention(dim_in) if i < 3 else SpatialTransformer(dim_in, num_heads_in, dim_head, depth=1, context_dim=context_dim)
            #     att_up = LinearAttention(dim_out) if i < 3 else SpatialTransformer(dim_out, num_heads_out, dim_head, depth=1, context_dim=context_dim)
            # else:
            #     att_down = LinearAttention(dim_in) # if i < 2 else Attention(dim_in)
            #     att_up = LinearAttention(dim_out) # if i < 2 else Attention(dim_out)

            if attn == "standard":
                if i in [3]:
                    att_module_down = AttentionBlock(dim_in, use_checkpoint=False, num_heads=num_heads_in,
                        num_head_channels=num_head_channels, use_new_attention_order=False)
                    att_module_up = AttentionBlock(dim_out, use_checkpoint=False, num_heads=num_heads_out,
                        num_head_channels=num_head_channels, use_new_attention_order=False)
                else:
                    att_down = LinearAttention(dim_in) # if i < 2 else Attention(dim_in)
                    att_up = LinearAttention(dim_out) # if i < 2 else Attention(dim_out)
                    att_module_down = Residual(PreNorm(dim_in, att_down))
                    att_module_up = Residual(PreNorm(dim_out, att_up))
            elif attn == "linear":
                att_down = LinearAttention(dim_in) # if i < 2 else Attention(dim_in)
                att_up = LinearAttention(dim_out) # if i < 2 else Attention(dim_out)
                att_module_down = Residual(PreNorm(dim_in, att_down))
                att_module_up = Residual(PreNorm(dim_out, att_up))
            elif attn == "none":
                att_module_down = LayerNorm(dim_in)
                att_module_up = LayerNorm(dim_out)
            else:
                raise ValueError(f"Unknown attn_type {attn_type}")

            self.downs.append(nn.ModuleList([
                block_class(dim_in=dim_in, dim_out=dim_in, time_emb_dim=time_dim),
                block_class(dim_in=dim_in, dim_out=dim_in, time_emb_dim=time_dim),

                # Residual(PreNorm(dim_in, att_down)) if attn else LayerNorm(dim_in),
                att_module_down,
                Downsample(dim_in, dim_out) if i != (self.depth-1) else default_conv(dim_in, dim_out)
            ]))

            self.ups.insert(0, nn.ModuleList([
                # First block
                block_class(dim_in=dim_out + dim_in, dim_out=dim_out, time_emb_dim=time_dim),
                # block_class(dim_in=dim_out + dim_in*2, dim_out=dim_out, time_emb_dim=time_dim),    # option:4 (decoder, concat embedding)

                # Second block
                block_class(dim_in=dim_out + dim_in, dim_out=dim_out, time_emb_dim=time_dim),    # option:1
                # block_class(dim_in=dim_out + dim_in*2, dim_out=dim_out, time_emb_dim=time_dim)   # option:2
                #     if self.cond_type == 'cond_module' else block_class(dim_in=dim_out + dim_in, dim_out=dim_out, time_emb_dim=time_dim),
                # block_class(dim_in=dim_out + dim_in, dim_out=dim_out, time_emb_dim=time_dim),    # option:3 (add embedding)
                # block_class(dim_in=dim_out + dim_in*2, dim_out=dim_out, time_emb_dim=time_dim),    # option:4 (decoder, concat embedding)

                # Residual(PreNorm(dim_out, att_up)) if attn else LayerNorm(dim_out),
                att_module_up,
                Upsample(dim_out, dim_in) if i!=0 else default_conv(dim_out, dim_in)
            ]))

            if cond_type == 'cond_module':
                self.cond_downs.append(nn.ModuleList([
                    block_class(dim_in=dim_in, dim_out=dim_in, time_emb_dim=time_dim),
                    Downsample(dim_in, dim_out) if i != (self.depth-1) else Identity()
                ]))
                self.SMblocks.append(SpatialModulationBlock(dim_in))
            if self.use_deg_embedding:
                self.residual_emb_downs.append(nn.ModuleList([
                    block_class(dim_in=dim_in, dim_out=dim_in),
                    Downsample(dim_in, dim_out) if i != (self.depth-1) else default_conv(dim_in, dim_out)
                ]))
        if self.use_deg_embedding:
            self.residual_emb_pooling = nn.AdaptiveAvgPool2d(1)
            self.residual_emb_mlp = nn.Sequential(
                nn.Linear(nf * ch_mult[-1], time_dim), 
                NonLinearity(),
                nn.Linear(time_dim, time_dim)
            )
        mid_dim = nf * ch_mult[-1]
        num_heads_mid = mid_dim // num_head_channels


        self.mid_block1 = block_class(dim_in=mid_dim, dim_out=mid_dim, time_emb_dim=time_dim)
        # self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim))) if attn else LayerNorm(mid_dim)
        # self.mid_attn = AttentionBlock(
        #                     mid_dim,
        #                     use_checkpoint=False,
        #                     num_heads=num_heads_out,
        #                     num_head_channels=num_head_channels,
        #                     use_new_attention_order=False,
        #                 )
        if attn == "standard":
            self.mid_attn = AttentionBlock(mid_dim, use_checkpoint=False, num_heads=num_heads_out,
                            num_head_channels=num_head_channels, use_new_attention_order=False)
        elif attn == "linear":
            self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim)))
        elif attn == "none":
            self.mid_attn = LayerNorm(mid_dim)
        self.mid_block2 = block_class(dim_in=mid_dim, dim_out=mid_dim, time_emb_dim=time_dim)

        self.final_res_block = block_class(dim_in=nf * 2, dim_out=nf, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(nf, out_nc, 3, 1, 1)

    def make_attention_block(
        i, dim_in, att_down, num_heads_in, num_head_channels,
        use_checkpoint=False, attn_type="none"
    ):
        if attn_type == "standard":
            return AttentionBlock(
                dim_in,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads_in,
                num_head_channels=num_head_channels,
                use_new_attention_order=False,
            )
        elif attn_type == "residual":
            return Residual(PreNorm(dim_in, att_down))
        elif attn_type == "none":
            # simplest placeholder (identity)
            return nn.Identity()
        else:
            raise ValueError(f"Unknown attn_type {attn_type}")

    def check_image_size(self, x, h, w):
        s = int(math.pow(2, self.depth))
        mod_pad_h = (s - h % s) % s
        mod_pad_w = (s - w % s) % s
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    # def forward(self, xt, cond, time, text_context=None, image_context=None):
    def forward(self, xt, mu, time, cond=None, text_context=None, image_context=None):

        if isinstance(time, int) or isinstance(time, float):
            time = torch.tensor([time]).to(xt.device)

        x = xt - mu
        residual_emb = cond - mu
        if self.cond_type == 'concat':
            x = torch.cat([x, mu, residual_emb], dim=1)
            # x = torch.cat([x, mu, cond], dim=1)
            # x = torch.cat([x, cond, mu-cond], dim=1)
            # x = torch.cat([x, mu, mu-cond], dim=1)
        else:
            x = torch.cat([x, mu], dim=1)

        H, W = x.shape[2:]
        x = self.check_image_size(x, H, W)
        residual_emb = self.check_image_size(residual_emb, H, W)

        x = self.init_conv(x)
        x_ = x.clone()

        t = self.time_mlp(time) 

        h = []
        for b1, b2, attn, downsample in self.downs:
            x = b1(x, t)
            h.append(x)

            x = b2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        cond_embedding = []

        if self.use_deg_embedding:
            residual_emb = self.init_conv_res_emb(residual_emb)
            for i, (b, downsample) in enumerate(self.residual_emb_downs):
                residual_emb = b(residual_emb)
                residual_emb = downsample(residual_emb)

            residual_emb = self.residual_emb_pooling(residual_emb)
            residual_emb = residual_emb.view(residual_emb.size(0), -1)
            residual_emb = self.residual_emb_mlp(residual_emb)

        if self.cond_type == 'cond_module':
            # cond = cond - mu
            cond = torch.cat([cond, mu], dim=1)
            cond = self.check_image_size(cond, H, W)
            
            cond = self.init_conv_cond(cond)
            n = len(self.cond_downs)
            for i, (b, downsample) in enumerate(self.cond_downs):
                cond = b(cond, t)
                # cond_embedding.append(cond)
                cond, mask = self.SMblocks[i](cond)
                if i == 0 and time.item() == 99:
                    mask = mask.squeeze()
                    mask = mask[1:-1, 1:-1]
                    mask = (mask - mask.min()) / (mask.max() - mask.min())
                    save_image(mask, "output.png")
                    # print(mask.shape)
                cond_embedding.append(cond)
                cond = downsample(cond)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for b1, b2, attn, upsample in self.ups:
            # x = torch.cat([x, h.pop()], dim=1)
            # x = b1(x, t)
            
            if self.cond_type == 'cond_module':
                cond_emb = cond_embedding.pop()
                # x = torch.cat([x, h.pop()+cond_emb], dim=1)
                x = torch.cat([x, h.pop(), cond_emb], dim=1)
                x = b1(x, t)
                # x = torch.cat([x, h.pop(), cond_embedding.pop()], dim=1)
                # x = torch.cat([x, h.pop()+cond_emb], dim=1)
                x = torch.cat([x, h.pop(), cond_emb], dim=1)
                x = b2(x, t)
            else:
                x = torch.cat([x, h.pop()], dim=1)
                x = b1(x, t)
                x = torch.cat([x, h.pop()], dim=1)
                if self.use_deg_embedding:
                    x = b2(x, residual_emb)
                else:
                    x = b2(x, t)

            x = attn(x)
            x = upsample(x)

        x = torch.cat([x, x_], dim=1)

        x = self.final_res_block(x, t)
        x = self.final_conv(x)

        x = x[..., :H, :W].contiguous()
        
        return x



