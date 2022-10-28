import math
from itertools import repeat
import collections.abc

import mindspore
from mindspore import nn
from mindspore import Tensor
from mindspore import Parameter

from mindspore import dtype as mstype
from mindspore import ops
from mindspore.common.initializer import Normal
from mindspore.common import initializer
from mindspore.ops import operations as P

import numpy as np
import mindspore.numpy as mind_np


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse
to_2tuple = _ntuple(2)


class Identity(nn.Cell):
    """Identity"""
    def construct(self, x):
        return x


class DropPath(nn.Cell):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, keep_prob=None, seed=0):
        super(DropPath, self).__init__()
        self.keep_prob = 1 - keep_prob
        seed = min(seed, 0)
        self.rand = P.UniformReal(seed=seed)
        self.shape = P.Shape()
        self.floor = P.Floor()

    def construct(self, x):
        if self.training:
            x_shape = self.shape(x)
            random_tensor = self.rand((x_shape[0], 1, 1))
            random_tensor = random_tensor + self.keep_prob
            random_tensor = self.floor(random_tensor)
            x = x / self.keep_prob
            x = x * random_tensor

        return x


class Mlp(nn.Cell):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super(Mlp,self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Dense(in_features, hidden_features, has_bias=True)
        self.act = act_layer()
        self.fc2 = nn.Dense(hidden_features, out_features, has_bias=True)
        self.drop = nn.Dropout(keep_prob = 1.0 - drop)

    def construct(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = ops.Reshape()(x, (B, H // window_size, window_size, W // window_size, window_size, C))
    windows = ops.Reshape()(ops.Transpose()(x, (0, 1, 3, 2, 4, 5)), (-1, window_size, window_size, C))
    return windows


def window_partition_noreshape(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (B, num_windows_h, num_windows_w, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    # pad feature maps to multiples of window size
    pad_l = pad_t = 0
    pad_r = (window_size - W % window_size) % window_size
    pad_b = (window_size - H % window_size) % window_size

    x = ops.Pad(((0, 0),(pad_l, pad_r),(pad_t, pad_b), (0, 0)))(x)
    B, H, W, C = x.shape
    x = ops.Reshape()(x, (B, H // window_size, window_size, W // window_size, window_size, C))
    windows = ops.Transpose()(x, (0, 1, 3, 2, 4, 5))
    return windows


def window_reverse(windows, window_size, H, W):

    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    
    B = windows.shape[0] / (H * W / window_size / window_size)
    x = ops.Reshape()(windows, (B, H // window_size, W // window_size, window_size, window_size, -1))
    x = ops.Transpose()(x, (0, 1, 3, 2, 4, 5))
    x = ops.Reshape()(x, (B, H, W, -1))
    return x


def get_topk_closest_indice(q_windows, k_windows, topk=1):
    # get pair-wise relative position index for each token inside the window
    coords_h_q = mind_np.arange(q_windows[0])
    coords_w_q = mind_np.arange(q_windows[1])
    
    if q_windows[0] != k_windows[0]:
        factor = k_windows[0] // q_windows[0]
        coords_h_q = coords_h_q * factor + factor // 2
        coords_w_q = coords_w_q * factor + factor // 2
    else:
        factor = 1 
    
    coords_q = ops.Stack()(mind_np.meshgrid(coords_h_q, coords_w_q))  # 2, Wh_q, Ww_q

    coords_h_k = mind_np.arange(k_windows[0])
    coords_w_k = mind_np.arange(k_windows[1])
    coords_k = ops.Stack()(mind_np.meshgrid(coords_h_k, coords_w_k))  # 2, Wh, Ww

    coords_flatten_q = ops.Flatten()(coords_q)  # 2, Wh_q*Ww_q
    coords_flatten_k = ops.Flatten()(coords_k)  # 2, Wh_k*Ww_k

    # relative_coords = coords_flatten_q[:, :, None] - coords_flatten_k[:, None, :]  # 2, Wh_q*Ww_q, Wh_k*Ww_k
    relative_coords = ops.expand_dims(coords_flatten_q, 2) - ops.expand_dims(coords_flatten_k, 1)
    relative_coords = ops.Cast()(relative_coords, mindspore.float32)

    relative_position_dists = ops.Sqrt()(relative_coords[0]**2 + relative_coords[1]**2)

    topk = min(topk, relative_position_dists.shape[1])

    _, topk_index_k = ops.TopK()(-relative_position_dists, topk)  # # B, topK, nHeads
    indice_topk = topk_index_k
    relative_coord_topk = ops.GatherD()(relative_coords, 2, ops.Tile()(ops.ExpandDims()(indice_topk, 0), (2, 1, 1)))
    return indice_topk, relative_coord_topk.transpose(1,2,0), topk


class WindowAttention(nn.Cell):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self,dim, input_resolution, expand_size, shift_size, window_size, window_size_glo, focal_window, 
                    focal_level, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., pool_method="none", topK=64):
        super().__init__()

        self.dim = dim
        self.shift_size = shift_size
        self.expand_size = expand_size
        self.window_size = window_size  # Wh, Ww
        self.window_size_glo = window_size_glo
        self.pool_method = pool_method
        self.input_resolution = input_resolution # NWh, NWw
        self.num_heads = num_heads
        head_dim = dim // num_heads        
        self.scale = Tensor(qk_scale or head_dim ** -0.5, mstype.float32)
        self.focal_level = focal_level
        self.focal_window = focal_window
        self.nWh, self.nWw = self.input_resolution[0] // self.window_size[0], self.input_resolution[1] // self.window_size[1]


        # get pair-wise relative position index for each token inside the window
        self.q = nn.Dense(in_channels=dim, out_channels=dim, has_bias=qkv_bias)
        self.k = nn.Dense(in_channels=dim, out_channels=dim, has_bias=qkv_bias)
        self.v = nn.Dense(in_channels=dim, out_channels=dim, has_bias=qkv_bias)

        self.attn_drop = nn.Dropout(keep_prob=1.0 - attn_drop)
        self.proj = nn.Dense(in_channels=dim, out_channels=dim, has_bias=True)
        self.proj_drop = nn.Dropout(keep_prob=1.0 - proj_drop)

        self.topK = topK
        
        coords_h_window = mind_np.arange(self.window_size[0]) - self.window_size[0] // 2
        coords_w_window = mind_np.arange(self.window_size[1]) - self.window_size[1] // 2        
        coords_window = ops.Stack( axis = -1)(mind_np.meshgrid(coords_h_window, coords_w_window))  # 2, Wh_q, Ww_q

        self.coord2rpb_all = nn.CellList()
    
        self.topks = []
        self.topk_cloest_indice = []
        self.topk_cloest_coords = []

        for k in range(self.focal_level):
            if k == 0:
                range_h = self.input_resolution[0]
                range_w = self.input_resolution[1]
            else:
                range_h = self.nWh
                range_w = self.nWw
            
            # build relative position range            
            topk_closest_indice, topk_closest_coord, topK_updated = get_topk_closest_indice(
                (self.nWh, self.nWw), (range_h, range_w), self.topK)
            self.topks.append(topK_updated)

            if k > 0:
                # scaling the coordinates for pooled windows
                topk_closest_coord = topk_closest_coord * self.window_size[0]
            topk_closest_coord_window = ops.ExpandDims()(topk_closest_coord, 1) + ops.Reshape()(coords_window, (-1, 2))[None, :, None, :]

            self.topk_cloest_indice.append(topk_closest_indice)
            self.topk_cloest_coords.append(ops.Cast()(topk_closest_coord_window, mindspore.float32))

            coord2rpb = nn.SequentialCell(
                nn.Dense(2, head_dim, has_bias=True),
                nn.ReLU(),
                nn.Dense(head_dim, self.num_heads, has_bias=True)
            )
            self.coord2rpb_all.append(coord2rpb)

    def construct(self, x_all, mask_all=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        x = x_all[0] # 
        B, nH, nW, C = x.shape
        q = ops.Reshape()(self.q(x), (B, nH, nW, C))
        k = ops.Reshape()(self.k(x), (B, nH, nW, C))
        v = ops.Reshape()(self.v(x), (B, nH, nW, C))

        # partition q map
        q_windows = window_partition(q, self.window_size[0])
        q_windows = ops.Reshape()(q_windows, (-1, self.window_size[0] * self.window_size[0], self.num_heads, C // self.num_heads))
        q_windows = ops.Transpose()(q_windows, (0, 2, 1, 3)) 

        k_all = []
        v_all = []
        topKs = []
        topk_rpbs = [] 

        for l_k in range(self.focal_level):
            topk_closest_indice = self.topk_cloest_indice[l_k]
            topk_indice_k = ops.Tile()(ops.Reshape()(topk_closest_indice, (1, -1)), (B, 1))
            topk_coords_k = self.topk_cloest_coords[l_k]

            topk_rpb_k = self.coord2rpb_all[l_k](topk_coords_k)
            topk_rpbs.append(topk_rpb_k)
            
            if l_k == 0:
                k_k = ops.Reshape()(k, (B, -1, self.num_heads, C // self.num_heads))
                v_k = ops.Reshape()(v, (B, -1, self.num_heads, C // self.num_heads))
            else:
                x_k = x_all[l_k]
                k_k = ops.Reshape()(self.k(x_k), (B, -1, self.num_heads, C // self.num_heads))
                v_k = ops.Reshape()(self.v(x_k), (B, -1, self.num_heads, C // self.num_heads))

            k_k_selected = ops.GatherD()(k_k, 1, ops.Tile()(ops.ExpandDims()(ops.Reshape()(topk_indice_k, (B, -1, 1)), -1), (1, 1, self.num_heads, C // self.num_heads)))
            v_k_selected = ops.GatherD()(v_k, 1, ops.Tile()(ops.ExpandDims()(ops.Reshape()(topk_indice_k, (B, -1, 1)), -1), (1, 1, self.num_heads, C // self.num_heads)))

            k_k_selected = ops.Transpose()(ops.Reshape()(k_k_selected, ((B,) + (topk_closest_indice.shape) + (self.num_heads, C // self.num_heads,))), (0, 1, 3, 2, 4))
            v_k_selected = ops.Transpose()(ops.Reshape()(v_k_selected, ((B,) + (topk_closest_indice.shape)+ (self.num_heads, C // self.num_heads,))), (0, 1, 3, 2, 4))

            k_all.append(ops.Reshape()(k_k_selected, (-1, self.num_heads, topk_closest_indice.shape[1], C // self.num_heads)))
            v_all.append(ops.Reshape()(v_k_selected, (-1, self.num_heads, topk_closest_indice.shape[1], C // self.num_heads)))

            topKs.append(topk_closest_indice.shape[1])

        k_all = ops.Concat(axis=2)(k_all)
        v_all = ops.Concat(axis=2)(v_all)

        q_windows = ops.Mul()(q_windows, self.scale)
        attn = ops.BatchMatMul(transpose_b = True)(q_windows, k_all)  

        topk_rpb_cat = ops.Transpose()(ops.Concat(axis=2)(topk_rpbs), (0, 3, 1, 2))
        topk_rpb_cat = ops.Tile()(topk_rpb_cat, (B, 1, 1, 1))
        topk_rpb_cat = ops.Reshape()(topk_rpb_cat, (attn.shape))

        attn = attn + topk_rpb_cat
        attn = nn.Softmax(axis = -1)(attn)
        attn = self.attn_drop(attn)

        x = ops.Reshape()(ops.Transpose()(ops.BatchMatMul()(attn, v_all), (0, 2, 1, 3)), (attn.shape[0],-1 ,C))

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

class FocalTransformerBlock(nn.Cell):
    r""" Focal Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, expand_size=0, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, pool_method="none",  
                 focal_level=1, focal_window=1, topK=64, use_layerscale=False, layerscale_value=1e-4):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.expand_size = expand_size
        self.mlp_ratio = mlp_ratio
        self.pool_method = pool_method
        self.focal_level = focal_level
        self.focal_window = focal_window
        self.use_layerscale = use_layerscale

        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.expand_size = 0
            # self.focal_level = 0
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.window_size_glo = self.window_size

        self.pool_layers = nn.CellList()
        if self.pool_method != "none":
            for k in range(self.focal_level-1):
                window_size_glo = self.window_size_glo
                if self.pool_method == "fc":
                    self.pool_layers.append(nn.Dense(in_channels = window_size_glo * window_size_glo, out_channels = 1, has_bias= True))
                    self.pool_layers[-1].weight.set_data(initializer.initializer(initializer.Constant(1./(window_size_glo * window_size_glo)), 
                                                self.pool_layers[-1].weight.shape,
                                                self.pool_layers[-1].weight.dtype))
                    self.pool_layers[-1].bias.set_data(initializer.initializer(initializer.Zero(), 
                                                self.pool_layers[-1].bias.shape,
                                                self.pool_layers[-1].bias.dtype))
                elif self.pool_method == "conv":
                    self.pool_layers.append(nn.Conv2d(dim, dim, kernel_size = window_size_glo, stride = window_size_glo, pad_mode = "valid", has_bias=True, group = dim))

        self.norm1 = norm_layer((dim,))
        self.attn = WindowAttention(
            dim, input_resolution=input_resolution, expand_size=self.expand_size, shift_size=self.shift_size, window_size=to_2tuple(self.window_size), 
            window_size_glo=to_2tuple(self.window_size_glo), focal_window=focal_window, 
            focal_level=self.focal_level, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, 
            pool_method=pool_method, topK=topK)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else Identity()
        self.norm2 = norm_layer((dim,))
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = np.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = ops.Reshape()(mask_windows, (-1, self.window_size * self.window_size))
            attn_mask = np.expand_dims(mask_windows, axis=1)  - np.expand_dims(mask_windows, axis=2)
            attn_mask = Tensor(np.where(attn_mask == 0, 0., -100.), dtype=mstype.float32)
            self.attn_mask = Parameter(attn_mask, requires_grad=False)
            self.roll_pos = Roll(self.shift_size)
            self.roll_neg = Roll(-self.shift_size)
        else:
            self.attn_mask = None

        if self.use_layerscale:
            self.gamma_1 = Parameter(layerscale_value * np.ones((dim)), requires_grad=True)
            self.gamma_2 = Parameter(layerscale_value * np.ones((dim)), requires_grad=True)

    def construct(self, x):
        H, W = self.input_resolution
        B, _, C = x.shape

        shortcut = x
        x = self.norm1(x)
        x = ops.Reshape()(x, (B, H, W, C))

        if self.shift_size > 0:
            shifted_x = self.roll_neg(x)
        else:
            shifted_x = x
        
        x_windows_all = [shifted_x]
        x_window_masks_all = [self.attn_mask]
        
        if self.focal_level > 1 and self.pool_method != "none": 
            # if we add coarser granularity and the pool method is not none
            for k in range(self.focal_level-1):     
                window_size_glo = self.window_size_glo 
                pooled_h = (H / self.window_size) * (2 ** k)
                pooled_w = (W / self.window_size) * (2 ** k)
                H_pool = pooled_h * window_size_glo
                W_pool = pooled_w * window_size_glo

                x_level_k = shifted_x
                # trim or pad shifted_x depending on the required size
                if H > H_pool:
                    trim_t = (H - H_pool) // 2
                    trim_b = H - H_pool - trim_t
                    x_level_k = x_level_k[:, trim_t:-trim_b]
                elif H < H_pool:
                    pad_t = (H_pool - H) // 2
                    pad_b = H_pool - H - pad_t
                    x_level_k = ops.Pad(((0, 0),(0, 0), (pad_t, pad_b), (0, 0)))(x_level_k)

                if W > W_pool:
                    trim_l = (W - W_pool) // 2
                    trim_r = W - W_pool - trim_l
                    x_level_k = x_level_k[:, :, trim_l:-trim_r]
                    
                elif W < W_pool:
                    pad_l = (W_pool - W) // 2
                    pad_r = W_pool - W - pad_l
                    x_level_k = ops.Pad(((0, 0),(pad_l,pad_r), (0, 0),(0, 0)))(x_level_k)

                x_windows_noreshape = window_partition_noreshape(x_level_k, window_size_glo) # B, nw, nw, window_size, window_size, C    
                nWh, nWw = x_windows_noreshape.shape[1:3]

                x_windows_pooled = None
                if self.pool_method == "mean":
                    x_windows_pooled = x_windows_noreshape.mean(axis=(3, 4)) # B, nWh, nWw, C
                elif self.pool_method == "max":
                    x_windows_pooled = ops.Reshape()(x_windows_noreshape.max(axis=(3, 4)), (B, nWh, nWw, C)) # B, nWh, nWw, C                    
                elif self.pool_method == "fc":
                    x_windows_noreshape = ops.Reshape()(x_windows_noreshape, (B, nWh, nWw, window_size_glo*window_size_glo, C))
                    x_windows_noreshape = ops.Transpose()(x_windows_noreshape, (0, 1, 2, 4, 3)) # B, nWh, nWw, C, wsize**2
                    x_windows_pooled = ops.Reshape()(self.pool_layers[k](x_windows_noreshape), (B, nWh, nWw, C)) # B, nWh, nWw, C                      
                elif self.pool_method == "conv":
                    x_windows_noreshape =ops.Transpose()(ops.Reshape()(x_windows_noreshape, (-1, window_size_glo, window_size_glo, C)), (0, 3, 1, 2)) # B * nw * nw, C, wsize, wsize
                    x_windows_pooled =ops.Reshape()(self.pool_layers[k](x_windows_noreshape), (B, nWh, nWw, C)) # B, nWh, nWw, C           

                x_windows_all += [x_windows_pooled]
                x_window_masks_all += [None]

        attn_windows = self.attn(x_windows_all, mask_all=x_window_masks_all)  # nW*B, window_size*window_size, C

        attn_windows = attn_windows[:, :self.window_size ** 2]
        
        # merge windows
        attn_windows = ops.Reshape()(attn_windows, (-1, self.window_size, self.window_size, C))

        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = self.roll_pos(shifted_x)
        else:
            x = shifted_x

        x =  ops.Reshape()(x, (B, H * W, C))
        # FFN
        x = shortcut + self.drop_path(x if (not self.use_layerscale) else (self.gamma_1 * x))
        x = x + self.drop_path(self.mlp(self.norm2(x)) if (not self.use_layerscale) else (self.gamma_2 * self.mlp(self.norm2(x))))
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"


class Roll(nn.Cell):
    """Roll Cell"""

    def __init__(self, shift_size, shift_axis=(1, 2)):
        super(Roll, self).__init__()
        self.shift_size = to_2tuple(shift_size)
        self.shift_axis = shift_axis

    def construct(self, x):
        x = mind_np.roll(x, self.shift_size, self.shift_axis)
        return x



class PatchMerging(nn.Cell):
    """ Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim[0] if isinstance(dim, tuple) and len(dim) == 1 else dim
        # Default False
        self.reduction = nn.Dense(in_channels=4 * dim, out_channels=2 * dim, has_bias=False)
        self.norm = norm_layer([dim * 4,])
        self.H, self.W = self.input_resolution
        self.H_2, self.W_2 = self.H // 2, self.W // 2
        self.H2W2 = int(self.H * self.W // 4)
        self.dim_mul_4 = int(dim * 4)
        self.H2W2 = int(self.H * self.W // 4)

    def construct(self, x):
        """
        x: B, H*W, C
        """
        B = x.shape[0]
        x = P.Reshape()(x, (B, self.H_2, 2, self.W_2, 2, self.dim))
        x = P.Transpose()(x, (0, 1, 3, 4, 2, 5))
        x = P.Reshape()(x, (B, self.H2W2, self.dim_mul_4))
        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"


class BasicLayer(nn.Cell):
    """ A basic Focal Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size, expand_size, expand_layer,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, pool_method="none", 
                 focal_level=1, focal_window=1, topK=64, use_conv_embed=False, use_shift=False, use_pre_norm=False, 
                 downsample=None, use_layerscale=False, layerscale_value=1e-4):
        super().__init__()
        
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        if expand_layer == "even":
            expand_factor = 0
        elif expand_layer == "odd":
            expand_factor = 1
        elif expand_layer == "all":
            expand_factor = -1

        # build blocks
        self.blocks = nn.CellList([
            FocalTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=(0 if (i % 2 == 0) else window_size // 2) if use_shift else 0,
                                 expand_size=0 if (i % 2 == expand_factor) else expand_size, 
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, 
                                 attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 pool_method=pool_method, 
                                 focal_level=focal_level, 
                                 focal_window=focal_window, 
                                 topK=topK, 
                                 use_layerscale=use_layerscale, 
                                 layerscale_value=layerscale_value)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(
                img_size=input_resolution, patch_size= 2, in_chans = dim, embed_dim = 2*dim, 
                use_conv_embed = use_conv_embed, norm_layer = norm_layer, use_pre_norm = use_pre_norm, 
                is_stem=False
            )
        else:
            self.downsample = None

    def construct(self, x):
        """construct"""
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = ops.Transpose()(ops.Reshape()(x , (x.shape[0], self.input_resolution[0], self.input_resolution[1], -1) ), (0, 3, 1, 2))
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"


class PatchEmbed(nn.Cell):
    """ Image to Patch Embedding

    Args:
        image_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Cell, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=(224, 224), patch_size=4, in_chans=3, embed_dim=96, use_conv_embed=False, norm_layer=None, use_pre_norm=False, is_stem=False):
        super().__init__()

        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.use_pre_norm = use_pre_norm
        self.use_conv_embed = use_conv_embed


        if use_conv_embed:
            # if we choose to use conv embedding, then we treat the stem and non-stem differently
            if is_stem:
                kernel_size = 7; padding = 2; stride = 4
            else:
                kernel_size = 3; padding = 1; stride = 2
            self.kernel_size = kernel_size
            self.proj = nn.Conv2d(in_channels=in_chans, out_channels=embed_dim, kernel_size = kernel_size, stride = stride,
                    pad_mode='pad',  padding = padding, has_bias=True)
        else:
            self.proj = nn.Conv2d(in_channels=in_chans, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size, pad_mode='valid', has_bias=True)
        
        if self.use_pre_norm:
            if norm_layer is not None:
                self.pre_norm = nn.GroupNorm(1, in_chans)
            else:
                self.pre_norm = None

        if norm_layer is not None:
            if isinstance(embed_dim, int):
                embed_dim = (embed_dim,)
            self.norm = norm_layer(embed_dim, epsilon=1e-5)
        else:
            self.norm = None

    def construct(self, x):
        """docstring"""
 
        B = x.shape[0]
        if self.use_pre_norm:
            x = self.pre_norm(x)

        # FIXME look at relaxing size constraints

        x = self.proj(x)
        x = ops.Reshape()(x, (B, self.embed_dim, -1)) 
        # B Ph*Pw C 
        x = ops.Transpose()(x, (0, 2, 1))  

        if self.norm is not None:
            x = self.norm(x)
        return x


class FocalTransformer(nn.Cell):
    r""" Focal Transformer: Focal Self-attention for Local-Global Interactions in Vision Transformer

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Focal Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False 
        use_shift (bool): Whether to use window shift proposed by Swin Transformer. We observe that using shift or not does not make difference to our Focal Transformer. Default: False
        focal_stages (list): Which stages to perform focal attention. Default: [0, 1, 2, 3], means all stages 
        focal_levels (list): How many focal levels at all stages. Note that this excludes the finest-grain level. Default: [1, 1, 1, 1] 
        focal_windows (list): The focal window size at all stages. Default: [7, 5, 3, 1] 
        expand_stages (list): Which stages to expand the finest grain window. Default: [0, 1, 2, 3], means all stages 
        expand_sizes (list): The expand size for the finest grain level. Default: [3, 3, 3, 3] 
        expand_layer (str): Which layers we want to expand the window for the finest grain leve. This can save computational and memory cost without the loss of performance. Default: "all" 
        use_conv_embed (bool): Whether use convolutional embedding. We noted that using convolutional embedding usually improve the performance, but we do not use it by default. Default: False 
        use_layerscale (bool): Whether use layerscale proposed in CaiT. Default: False 
        layerscale_value (float): Value for layer scale. Default: 1e-4 
        use_pre_norm (bool): Whether use pre-norm in patch merging/embedding layer to control the feature magtigute. Default: False
    """
    def __init__(self, 
                img_size=224, 
                patch_size=4, 
                in_chans=3, 
                num_classes=1000,
                embed_dim=96, 
                depths=[2, 2, 6, 2], 
                num_heads=[3, 6, 12, 24],
                window_size=7, 
                mlp_ratio=4., 
                qkv_bias=True, 
                qk_scale=None,
                drop_rate=0., 
                attn_drop_rate=0., 
                drop_path_rate=0.1,
                norm_layer=nn.LayerNorm, 
                ape=False, 
                patch_norm=True,
                use_checkpoint=False,                 
                use_shift=False, 
                focal_stages=[0, 1, 2, 3], 
                focal_levels=[1, 1, 1, 1], 
                focal_windows=[7, 5, 3, 1], 
                focal_topK=64, 
                focal_pool="fc", 
                expand_stages=[0, 1, 2, 3], 
                expand_sizes=[3, 3, 3, 3],
                expand_layer="all", 
                use_conv_embed=False, 
                use_layerscale=False, 
                layerscale_value=1e-4, 
                use_pre_norm=False, 
                **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        
        # split image into patches using either non-overlapped embedding or overlapped embedding
        self.patch_embed = PatchEmbed(
            img_size=to_2tuple(img_size), patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, use_conv_embed=use_conv_embed, is_stem=True, 
            norm_layer=norm_layer if self.patch_norm else None)

        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:  
            self.absolute_pos_embed = Parameter(Tensor(np.zeros(1, num_patches, embed_dim)), mindspore.float32)
 
        self.pos_drop = nn.Dropout(keep_prob = 1.0 - drop_rate)

        # stochastic depth
        dpr = [x.item() for x in np.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.CellList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, 
                               qk_scale=qk_scale,
                               drop=drop_rate, 
                               attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer, 
                               pool_method=focal_pool if i_layer in focal_stages else "none",
                               downsample=PatchEmbed if (i_layer < self.num_layers - 1) else None,
                               focal_level=focal_levels[i_layer], 
                               focal_window=focal_windows[i_layer], 
                               topK=focal_topK, 
                               expand_size=expand_sizes[i_layer], 
                               expand_layer=expand_layer,                           
                               use_conv_embed=use_conv_embed,
                               use_shift=use_shift, 
                               use_pre_norm=use_pre_norm, 
                               use_layerscale=use_layerscale, 
                               layerscale_value=layerscale_value)
            self.layers.append(layer)

        self.norm = norm_layer((self.num_features,))
        self.avgpool = P.ReduceMean(keep_dims=False)
        self.head = nn.Dense(in_channels = self.num_features, out_channels = num_classes, has_bias= True) if num_classes > 0 else Identity()
        self.init_weights()

    def init_weights(self):
        """
        init_weights

        """
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Dense):
                cell.weight.set_data(initializer.initializer(initializer.TruncatedNormal(sigma=0.02), 
                                                             cell.weight.shape,
                                                             cell.weight.dtype))
                if isinstance(cell, nn.Dense) and cell.bias is not None:
                    cell.bias.set_data(initializer.initializer(initializer.Zero(),
                                                               cell.bias.shape,
                                                               cell.bias.dtype))
            
            elif isinstance(cell, nn.Conv2d):
                fan_out = cell.kernel_size[0] * cell.kernel_size[1] * cell.out_channels
                fan_out //= cell.group
                cell.weight.set_data(initializer.initializer(initializer.Normal(sigma=math.sqrt(2.0 / fan_out)),
                                                             cell.weight.shape,
                                                             cell.weight.dtype))
                                                             
                if isinstance(cell, nn.Conv2d) and cell.bias is not None:
                    cell.bias.set_data(initializer.initializer(initializer.Zero(),
                                                               cell.bias.shape,
                                                               cell.bias.dtype))      

            elif isinstance(cell, nn.LayerNorm):
                cell.gamma.set_data(initializer.initializer(initializer.One(),
                                                            cell.gamma.shape,
                                                            cell.gamma.dtype))
                cell.beta.set_data(initializer.initializer(initializer.Zero(),
                                                           cell.beta.shape,
                                                           cell.beta.dtype))


    def no_weight_decay(self):
        return {'absolute_pos_embed'}


    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)  # B L C
        x = self.avgpool(ops.Transpose()(x, (0, 2, 1)), 2)  # B C
        return x

    def construct(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def small_focal(**kwargs):
    img_size = 224
    model = FocalTransformer(
        img_size=img_size, 
        use_conv_embed=True, 
        embed_dim=96, 
        depths=[2,2,18,2], 
        num_heads=[3,6,12,24],
        window_size=7,
        focal_pool = "fc",
        focal_stages=[0, 1, 2, 3],
        focal_levels=[2,2,2,2],
        focal_windows=[7,5,3,1], 
        expand_sizes=[3,3,3,3],
        expand_layer="all", 
        use_shift=False, 
        focal_topK = 128,
        **kwargs
    )

    return model

if __name__ == '__main__':
    from mindspore import context
    mode = {
        0: context.GRAPH_MODE,
        1: context.PYNATIVE_MODE
    }
    graph_mode = 0
    context.set_context(mode=mode[graph_mode], device_target="GPU")

    img_size = 224
    x = Tensor(np.ones([1, 3, 224,224]), mindspore.float32)

    # focal small
    model = FocalTransformer(
        img_size=img_size, 
        use_conv_embed=True, 
        embed_dim=96, 
        depths=[2,2,18,2], 
        num_heads=[3,6,12,24],
        window_size=7,
        focal_pool = "fc",
        focal_stages=[0, 1, 2, 3],
        focal_levels=[2,2,2,2],
        focal_windows=[7,5,3,1], 
        expand_sizes=[3,3,3,3],
        expand_layer="all", 
        drop_path_rate = 0.3, 
        use_shift=False, 
        focal_topK = 128,
    )

    n_parameters = sum(ops.Size()(p) for p in model.get_parameters() if p.requires_grad)

    print("parameters: ", n_parameters)

    y = model(x)
    print(y.shape)