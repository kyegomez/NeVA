from collections import namedtuple
from dataclasses import dataclass
from functools import partial, wraps

import torch
import torch.nn.functional as F
from einops import rearrange
from packaging import version
from torch import Tensor, einsum, nn

#constants
EfficientAttentionConfig = namedtuple("EfficientAttentionConfig", ['enable_flash', 'enable_math', 'enable_mem_efficient'])


@dataclass
class Intermediates:
    qk_similarities: Tensor = None
    pre_softmax_attn: Tensor = None
    post_softmax_attn: Tensor = None

    def to_tuple(self):
        return (self.qk_similarities, self.pre_softmax_attn, self.post_softmax_attn)
    

#helpers
def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def compact(arr):
    return [*filter(exists, arr)]


def once(fn):
    called = False
    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)
    return inner

print_once = once(print)

def create_casual_mask(i, j, device):
    return torch.ones((i, j), device=device, dtype=torch.bool).triu(j - i + 1)

def onnx_create_casual_mask(i, j, device):
    r = torch.arange(i, device=device)
    causal_mask = rearrange(r, 'i -> i 1') < rearrange(r, 'j -> 1 j')
    causal_mask = F.pad(causal_mask, (j - 1, 0), value=False)
    return causal_mask

#main clas

class Attend(nn.Module):
    def __init__(self,
                 *,
                 dropout=0.,
                 causal=False,
                 heads=None,
                 scale=None,
                 qk_norm=False,
                 flash=False,
                 add_zero_kv=False,
                 onnxable=False):
        super().__init__()
        self.scale = scale
        self.qk_norm = qk_norm
        self.causal = causal

        self.create_causal_mask = onnx_create_casual_mask if onnxable else create_casual_mask
        self.attn_fn = partial(F.softmax, dtype=torch.float32) if not qk_norm else F.softmax
        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)

        self.add_zero_kv = add_zero_kv

        self.flash = flash

        #determine efficient configs
        self.cpu_config = EfficientAttentionConfig(True, True, True)
        self.cuda_config = None
        
        if not torch.cuda.is_available() or not flash:
            return
        
        self.cuda_config = EfficientAttentionConfig(True, False, False)
    
    def flash_attn(
            self,
            q,
            k,
            v,
            mask=None,
            attn_bias=None
    ):
        batch, heads, q_len, _, k_len, is_cuda, device = *q.shape, k.shape[-2], q.is_cuda, q.device

        if k.ndim == 3:
            k = rearrange(k, 'b ... -> b 1 ...').expand_as(q)
        
        if v.ndim == 3:
            v = rearrange(v, 'b ... -> b 1 ...').expand_as(q)

        #handle scale
        if self.qk_norm:
            default_scale = q.shape[-1] ** -0.5
            q = q * (default_scale / self.scale)
        
        causal = self.causal

        if exists(mask):
            assert mask.ndim == 4
            mask = mask.expand(batch, heads, q_len, k_len)

            if causal:
                causal_mask = self.create_causal_mask(q_len, k_len, device=device)
                mask = mask & ~ causal_mask
                causal=False

        #alibi
        if exists(attn_bias):
            attn_bias = rearrange(attn_bias, 'h i j -> 1 h i j').expand(batch, heads, -1, -1)

            mask_value = -torch.finfo(q.dtype).max
            
            if exists(mask):
                attn_bias = attn_bias.masked_fill(~mask, mask_value // 2)
            elif causal:
                causal_mask = self.create_causal_mask(q_len, k_len, device=device)
                attn_bias = attn_bias.masked_fill(causal_mask, mask_value // 2)
                causal = False

            
            mask = attn_bias

        config = self.cuda_config if is_cuda else self.cpu_config

        with torch.backends.cuda.sdk_kernel(**config._asdict()):
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask,
                dropout_p=self.dropout if self.training else 0.,
                is_causal=causal
            )
            return out, Intermediates()
        
    def forward(
            self,
            q,
            k,
            v,
            mask=None,
            attn_bias=None,
            prev_attn=None
    ):
        n, device = q.shape[-2], q.device
        scale = default(self.scale, q.shape[-1] ** -0.5)

        if self.add_zero_kv:
            k, v = map(lambda t: F.pad(t, (0, 0, 1, 0), value=0.), (k, v))

            if exists(mask):
                mask = F.pad(mask, (1, 0), value=True)
            
            if exists(attn_bias):
                attn_bias = F.pad(attn_bias, (1, 0), value=0.)
        
        if self.flash:
            assert not exists(prev_attn), 'residual attention not compatible with flash'
            return self.flash_attn(q, k, v, mask=mask, attn_bias=attn_bias)
        
        kv_einsum_eq = 'b j d' if k.ndim == 3 else 'b h j d'
        dots = einsum(f'b h i d {kv_einsum_eq} -> b h i j', q, k) * scale

        if exists(prev_attn):
            dots = dots + prev_attn
        
        qk_similarities = dots.clone()

        if self.talking_heads:
            dots = self.pre_softmax_talking_heads(dots)
        
        if exists(attn_bias):
            dots = dots + attn_bias
        
        i, j, dtype = *dots.shape[-2:], dots.dtype
        pre_softmax_attn = dots.clone()

        mask_value = -torch.finfo(dots.dtype).max

        if exists(mask):
            dots = dots.masked_fill(~mask, mask_value)
        
        if self.causal:
            causal_mask = self.creat_causal_mask(i, j, device=device)
            dots = dots.masked_fill(causal_mask, mask_value)
        
        attn = self.attn_fn(dots, dim=-1)
        attn = attn.type(dtype)
        post_softmax_attn = attn.clone()

        attn = self.attn_dropout(attn)

        out = einsum(f'b h i j, {kv_einsum_eq} -> b h i d', attn, v)

        intermediates = Intermediates(
            qk_similarities=qk_similarities,
            pre_softmax_attn=pre_softmax_attn,
            post_softmax_attn=post_softmax_attn
        )
        
        return out, intermediates