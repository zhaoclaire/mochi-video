from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

import mochi_preview.dit.joint_model.context_parallel as cp
from mochi_preview.vae.cp_conv import cp_pass_frames, gather_all_frames


def cast_tuple(t, length=1):
    return t if isinstance(t, tuple) else ((t,) * length)


class GroupNormSpatial(nn.GroupNorm):
    """
    GroupNorm applied per-frame.
    """

    def forward(self, x: torch.Tensor, *, chunk_size: int = 8):
        B, C, T, H, W = x.shape
        x = rearrange(x, "B C T H W -> (B T) C H W")
        # Run group norm in chunks.
        output = torch.empty_like(x)
        for b in range(0, B * T, chunk_size):
            output[b : b + chunk_size] = super().forward(x[b : b + chunk_size])
        return rearrange(output, "(B T) C H W -> B C T H W", B=B, T=T)


class SafeConv3d(torch.nn.Conv3d):
    """
    NOTE: No support for padding along time dimension.
          Input must already be padded along time.
    """

    def forward(self, input):
        memory_count = torch.prod(torch.tensor(input.shape)).item() * 2 / 1024**3
        if memory_count > 2:
            part_num = int(memory_count / 2) + 1
            k = self.kernel_size[0]
            input_idx = torch.arange(k - 1, input.size(2))
            input_chunks_idx = torch.chunk(input_idx, part_num, dim=0)

            # assert self.kernel_size == (3, 3, 3), f"kernel_size {self.kernel_size} != (3, 3, 3)"
            assert self.stride[0] == 1, f"stride {self.stride}"
            assert self.dilation[0] == 1, f"dilation {self.dilation}"
            assert self.padding[0] == 0, f"padding {self.padding}"

            # Comptue output size
            assert not input.requires_grad
            B, _, T_in, H_in, W_in = input.shape
            output_size = (
                B,
                self.out_channels,
                T_in - k + 1,
                H_in // self.stride[1],
                W_in // self.stride[2],
            )
            output = torch.empty(output_size, dtype=input.dtype, device=input.device)
            for input_chunk_idx in input_chunks_idx:
                input_s = input_chunk_idx[0] - k + 1
                input_e = input_chunk_idx[-1] + 1
                input_chunk = input[:, :, input_s:input_e, :, :]
                output_chunk = super(SafeConv3d, self).forward(input_chunk)

                output_s = input_s
                output_e = output_s + output_chunk.size(2)
                output[:, :, output_s:output_e, :, :] = output_chunk

            return output
        else:
            return super(SafeConv3d, self).forward(input)


class StridedSafeConv3d(torch.nn.Conv3d):
    def forward(self, input, local_shard: bool = False):
        assert self.stride[0] == self.kernel_size[0]
        assert self.dilation[0] == 1
        assert self.padding[0] == 0

        kernel_size = self.kernel_size[0]
        stride = self.stride[0]
        T_in = input.size(2)
        T_out = T_in // kernel_size

        # Parallel implementation.
        if local_shard:
            idx = torch.arange(T_out)
            idx = cp.local_shard(idx, dim=0)
            start = idx.min() * stride
            end = idx.max() * stride + kernel_size
            local_input = input[:, :, start:end, :, :]
            return torch.nn.Conv3d.forward(self, local_input)

        raise NotImplementedError


class ContextParallelConv3d(SafeConv3d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int]],
        causal: bool = True,
        context_parallel: bool = True,
        **kwargs,
    ):
        self.causal = causal
        self.context_parallel = context_parallel
        kernel_size = cast_tuple(kernel_size, 3)
        stride = cast_tuple(stride, 3)
        height_pad = (kernel_size[1] - 1) // 2
        width_pad = (kernel_size[2] - 1) // 2

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=(1, 1, 1),
            padding=(0, height_pad, width_pad),
            **kwargs,
        )

    def forward(self, x: torch.Tensor):
        cp_rank, cp_world_size = cp.get_cp_rank_size()

        # Compute padding amounts.
        context_size = self.kernel_size[0] - 1
        if self.causal:
            pad_front = context_size
            pad_back = 0
        else:
            pad_front = context_size // 2
            pad_back = context_size - pad_front

        # Apply padding.
        assert self.padding_mode == "replicate"  # DEBUG
        mode = "constant" if self.padding_mode == "zeros" else self.padding_mode
        if self.context_parallel and cp_world_size == 1:
            x = F.pad(x, (0, 0, 0, 0, pad_front, pad_back), mode=mode)
        else:
            if cp_rank == 0:
                x = F.pad(x, (0, 0, 0, 0, pad_front, 0), mode=mode)
            elif cp_rank == cp_world_size - 1 and pad_back:
                x = F.pad(x, (0, 0, 0, 0, 0, pad_back), mode=mode)

        if self.context_parallel and cp_world_size == 1:
            return super().forward(x)

        if self.stride[0] == 1:
            # Receive some frames from previous rank.
            x = cp_pass_frames(x, context_size)
            return super().forward(x)

        # Less efficient implementation for strided convs.
        # All gather x, infer and chunk.
        assert (
            x.dtype == torch.bfloat16
        ), f"Expected x to be of type torch.bfloat16, got {x.dtype}"

        x = gather_all_frames(x)  # [B, C, k - 1 + global_T, H, W]
        return StridedSafeConv3d.forward(self, x, local_shard=True)


class Conv1x1(nn.Linear):
    """*1x1 Conv implemented with a linear layer."""

    def __init__(self, in_features: int, out_features: int, *args, **kwargs):
        super().__init__(in_features, out_features, *args, **kwargs)

    def forward(self, x: torch.Tensor):
        """Forward pass.

        Args:
            x: Input tensor. Shape: [B, C, *] or [B, *, C].

        Returns:
            x: Output tensor. Shape: [B, C', *] or [B, *, C'].
        """
        x = x.movedim(1, -1)
        x = super().forward(x)
        x = x.movedim(-1, 1)
        return x


class DepthToSpaceTime(nn.Module):
    def __init__(
        self,
        temporal_expansion: int,
        spatial_expansion: int,
    ):
        super().__init__()
        self.temporal_expansion = temporal_expansion
        self.spatial_expansion = spatial_expansion

    # When printed, this module should show the temporal and spatial expansion factors.
    def extra_repr(self):
        return f"texp={self.temporal_expansion}, sexp={self.spatial_expansion}"

    def forward(self, x: torch.Tensor):
        """Forward pass.

        Args:
            x: Input tensor. Shape: [B, C, T, H, W].

        Returns:
            x: Rearranged tensor. Shape: [B, C/(st*s*s), T*st, H*s, W*s].
        """
        x = rearrange(
            x,
            "B (C st sh sw) T H W -> B C (T st) (H sh) (W sw)",
            st=self.temporal_expansion,
            sh=self.spatial_expansion,
            sw=self.spatial_expansion,
        )

        cp_rank, _ = cp.get_cp_rank_size()
        if self.temporal_expansion > 1 and cp_rank == 0:
            # Drop the first self.temporal_expansion - 1 frames.
            # This is because we always want the 3x3x3 conv filter to only apply
            # to the first frame, and the first frame doesn't need to be repeated.
            assert all(x.shape)
            x = x[:, :, self.temporal_expansion - 1 :]
            assert all(x.shape)

        return x


def norm_fn(
    in_channels: int,
    affine: bool = True,
):
    return GroupNormSpatial(affine=affine, num_groups=32, num_channels=in_channels)


class ResBlock(nn.Module):
    """Residual block that preserves the spatial dimensions."""

    def __init__(
        self,
        channels: int,
        *,
        affine: bool = True,
        attn_block: Optional[nn.Module] = None,
        padding_mode: str = "replicate",
        causal: bool = True,
    ):
        super().__init__()
        self.channels = channels

        assert causal
        self.stack = nn.Sequential(
            norm_fn(channels, affine=affine),
            nn.SiLU(inplace=True),
            ContextParallelConv3d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=(3, 3, 3),
                stride=(1, 1, 1),
                padding_mode=padding_mode,
                bias=True,
                causal=causal,
            ),
            norm_fn(channels, affine=affine),
            nn.SiLU(inplace=True),
            ContextParallelConv3d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=(3, 3, 3),
                stride=(1, 1, 1),
                padding_mode=padding_mode,
                bias=True,
                causal=causal,
            ),
        )

        self.attn_block = attn_block if attn_block else nn.Identity()

    def forward(self, x: torch.Tensor):
        """Forward pass.

        Args:
            x: Input tensor. Shape: [B, C, T, H, W].
        """
        residual = x
        x = self.stack(x)
        x = x + residual
        del residual

        return self.attn_block(x)


def prepare_for_attention(qkv: torch.Tensor, head_dim: int, qk_norm: bool = True):
    """Prepare qkv tensor for attention and normalize qk.

    Args:
        qkv: Input tensor. Shape: [B, L, 3 * num_heads * head_dim].

    Returns:
        q, k, v: qkv tensor split into q, k, v. Shape: [B, num_heads, L, head_dim].
    """
    assert qkv.ndim == 3  # [B, L, 3 * num_heads * head_dim]
    assert qkv.size(2) % (3 * head_dim) == 0
    num_heads = qkv.size(2) // (3 * head_dim)
    qkv = qkv.unflatten(2, (3, num_heads, head_dim))

    q, k, v = qkv.unbind(2)  # [B, L, num_heads, head_dim]
    q = q.transpose(1, 2)  # [B, num_heads, L, head_dim]
    k = k.transpose(1, 2)  # [B, num_heads, L, head_dim]
    v = v.transpose(1, 2)  # [B, num_heads, L, head_dim]

    if qk_norm:
        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)

        # Mixed precision can change the dtype of normed q/k to float32.
        q = q.to(dtype=qkv.dtype)
        k = k.to(dtype=qkv.dtype)

    return q, k, v


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        head_dim: int = 32,
        qkv_bias: bool = False,
        out_bias: bool = True,
        qk_norm: bool = True,
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.num_heads = dim // head_dim
        self.qk_norm = qk_norm

        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.out = nn.Linear(dim, dim, bias=out_bias)

    def forward(
        self,
        x: torch.Tensor,
        *,
        chunk_size=2**15,
    ) -> torch.Tensor:
        """Compute temporal self-attention.

        Args:
            x: Input tensor. Shape: [B, C, T, H, W].
            chunk_size: Chunk size for large tensors.

        Returns:
            x: Output tensor. Shape: [B, C, T, H, W].
        """
        B, _, T, H, W = x.shape

        if T == 1:
            # No attention for single frame.
            x = x.movedim(1, -1)  # [B, C, T, H, W] -> [B, T, H, W, C]
            qkv = self.qkv(x)
            _, _, x = qkv.chunk(3, dim=-1)  # Throw away queries and keys.
            x = self.out(x)
            return x.movedim(-1, 1)  # [B, T, H, W, C] -> [B, C, T, H, W]

        # 1D temporal attention.
        x = rearrange(x, "B C t h w -> (B h w) t C")
        qkv = self.qkv(x)

        # Input: qkv with shape [B, t, 3 * num_heads * head_dim]
        # Output: x with shape [B, num_heads, t, head_dim]
        q, k, v = prepare_for_attention(qkv, self.head_dim, qk_norm=self.qk_norm)

        attn_kwargs = dict(
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
            scale=self.head_dim**-0.5,
        )

        if q.size(0) <= chunk_size:
            x = F.scaled_dot_product_attention(
                q, k, v, **attn_kwargs
            )  # [B, num_heads, t, head_dim]
        else:
            # Evaluate in chunks to avoid `RuntimeError: CUDA error: invalid configuration argument.`
            # Chunks of 2**16 and up cause an error.
            x = torch.empty_like(q)
            for i in range(0, q.size(0), chunk_size):
                qc = q[i : i + chunk_size]
                kc = k[i : i + chunk_size]
                vc = v[i : i + chunk_size]
                chunk = F.scaled_dot_product_attention(qc, kc, vc, **attn_kwargs)
                x[i : i + chunk_size].copy_(chunk)

        assert x.size(0) == q.size(0)
        x = x.transpose(1, 2)  # [B, t, num_heads, head_dim]
        x = x.flatten(2)  # [B, t, num_heads * head_dim]

        x = self.out(x)
        x = rearrange(x, "(B h w) t C -> B C t h w", B=B, h=H, w=W)
        return x


class AttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        **attn_kwargs,
    ) -> None:
        super().__init__()
        self.norm = norm_fn(dim)
        self.attn = Attention(dim, **attn_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.attn(self.norm(x))


class CausalUpsampleBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_res_blocks: int,
        *,
        temporal_expansion: int = 2,
        spatial_expansion: int = 2,
        **block_kwargs,
    ):
        super().__init__()

        blocks = []
        for _ in range(num_res_blocks):
            blocks.append(block_fn(in_channels, **block_kwargs))
        self.blocks = nn.Sequential(*blocks)

        self.temporal_expansion = temporal_expansion
        self.spatial_expansion = spatial_expansion

        # Change channels in the final convolution layer.
        self.proj = Conv1x1(
            in_channels,
            out_channels * temporal_expansion * (spatial_expansion**2),
        )

        self.d2st = DepthToSpaceTime(
            temporal_expansion=temporal_expansion, spatial_expansion=spatial_expansion
        )

    def forward(self, x):
        x = self.blocks(x)
        x = self.proj(x)
        x = self.d2st(x)
        return x


def block_fn(channels, *, has_attention: bool = False, **block_kwargs):
    attn_block = AttentionBlock(channels) if has_attention else None

    return ResBlock(
        channels, affine=True, attn_block=attn_block, **block_kwargs
    )


class DownsampleBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_res_blocks,
        *,
        temporal_reduction=2,
        spatial_reduction=2,
        **block_kwargs,
    ):
        """
        Downsample block for the VAE encoder.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            num_res_blocks: Number of residual blocks.
            temporal_reduction: Temporal reduction factor.
            spatial_reduction: Spatial reduction factor.
        """
        super().__init__()
        layers = []

        # Change the channel count in the strided convolution.
        # This lets the ResBlock have uniform channel count,
        # as in ConvNeXt.
        assert in_channels != out_channels
        layers.append(
            ContextParallelConv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(temporal_reduction, spatial_reduction, spatial_reduction),
                stride=(temporal_reduction, spatial_reduction, spatial_reduction),
                padding_mode="replicate",
                bias=True,
            )
        )

        for _ in range(num_res_blocks):
            layers.append(block_fn(out_channels, **block_kwargs))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


def add_fourier_features(inputs: torch.Tensor, start=6, stop=8, step=1):
    num_freqs = (stop - start) // step
    assert inputs.ndim == 5
    C = inputs.size(1)

    # Create Base 2 Fourier features.
    freqs = torch.arange(start, stop, step, dtype=inputs.dtype, device=inputs.device)
    assert num_freqs == len(freqs)
    w = torch.pow(2.0, freqs) * (2 * torch.pi)  # [num_freqs]
    C = inputs.shape[1]
    w = w.repeat(C)[None, :, None, None, None]  # [1, C * num_freqs, 1, 1, 1]

    # Interleaved repeat of input channels to match w.
    h = inputs.repeat_interleave(num_freqs, dim=1)  # [B, C * num_freqs, T, H, W]
    # Scale channels by frequency.
    h = w * h

    return torch.cat(
        [
            inputs,
            torch.sin(h),
            torch.cos(h),
        ],
        dim=1,
    )


class FourierFeatures(nn.Module):
    def __init__(self, start: int = 6, stop: int = 8, step: int = 1):
        super().__init__()
        self.start = start
        self.stop = stop
        self.step = step

    def forward(self, inputs):
        """Add Fourier features to inputs.

        Args:
            inputs: Input tensor. Shape: [B, C, T, H, W]

        Returns:
            h: Output tensor. Shape: [B, (1 + 2 * num_freqs) * C, T, H, W]
        """
        return add_fourier_features(inputs, self.start, self.stop, self.step)


class Decoder(nn.Module):
    def __init__(
        self,
        *,
        out_channels: int = 3,
        latent_dim: int,
        base_channels: int,
        channel_multipliers: List[int],
        num_res_blocks: List[int],
        temporal_expansions: Optional[List[int]] = None,
        spatial_expansions: Optional[List[int]] = None,
        has_attention: List[bool],
        output_norm: bool = True,
        nonlinearity: str = "silu",
        output_nonlinearity: str = "silu",
        causal: bool = True,
        **block_kwargs,
    ):
        super().__init__()
        self.input_channels = latent_dim
        self.base_channels = base_channels
        self.channel_multipliers = channel_multipliers
        self.num_res_blocks = num_res_blocks
        self.output_nonlinearity = output_nonlinearity
        assert nonlinearity == "silu"
        assert causal

        ch = [mult * base_channels for mult in channel_multipliers]
        self.num_up_blocks = len(ch) - 1
        assert len(num_res_blocks) == self.num_up_blocks + 2

        blocks = []

        first_block = [
            nn.Conv3d(latent_dim, ch[-1], kernel_size=(1, 1, 1))
        ]  # Input layer.
        # First set of blocks preserve channel count.
        for _ in range(num_res_blocks[-1]):
            first_block.append(
                block_fn(
                    ch[-1],
                    has_attention=has_attention[-1],
                    causal=causal,
                    **block_kwargs,
                )
            )
        blocks.append(nn.Sequential(*first_block))

        assert len(temporal_expansions) == len(spatial_expansions) == self.num_up_blocks
        assert len(num_res_blocks) == len(has_attention) == self.num_up_blocks + 2

        upsample_block_fn = CausalUpsampleBlock

        for i in range(self.num_up_blocks):
            block = upsample_block_fn(
                ch[-i - 1],
                ch[-i - 2],
                num_res_blocks=num_res_blocks[-i - 2],
                has_attention=has_attention[-i - 2],
                temporal_expansion=temporal_expansions[-i - 1],
                spatial_expansion=spatial_expansions[-i - 1],
                causal=causal,
                **block_kwargs,
            )
            blocks.append(block)

        assert not output_norm

        # Last block. Preserve channel count.
        last_block = []
        for _ in range(num_res_blocks[0]):
            last_block.append(
                block_fn(
                    ch[0], has_attention=has_attention[0], causal=causal, **block_kwargs
                )
            )
        blocks.append(nn.Sequential(*last_block))

        self.blocks = nn.ModuleList(blocks)
        self.output_proj = Conv1x1(ch[0], out_channels)

    def forward(self, x):
        """Forward pass.

        Args:
            x: Latent tensor. Shape: [B, input_channels, t, h, w]. Scaled [-1, 1].

        Returns:
            x: Reconstructed video tensor. Shape: [B, C, T, H, W]. Scaled to [-1, 1].
               T + 1 = (t - 1) * 4.
               H = h * 16, W = w * 16.
        """
        for block in self.blocks:
            x = block(x)

        if self.output_nonlinearity == "silu":
            x = F.silu(x, inplace=not self.training)
        else:
            assert (
                not self.output_nonlinearity
            )  # StyleGAN3 omits the to-RGB nonlinearity.

        return self.output_proj(x).contiguous()


def make_broadcastable(
    tensor: torch.Tensor,
    axis: int,
    ndim: int,
) -> torch.Tensor:
    """
    Reshapes the input tensor to have singleton dimensions in all axes except the specified axis.

    Args:
        tensor (torch.Tensor): The tensor to reshape. Typically 1D.
        axis (int): The axis along which the tensor should retain its original size.
        ndim (int): The total number of dimensions the reshaped tensor should have.

    Returns:
        torch.Tensor: The reshaped tensor with shape suitable for broadcasting.
    """
    if tensor.dim() != 1:
        raise ValueError(f"Expected tensor to be 1D, but got {tensor.dim()}D tensor.")

    axis = (axis + ndim) % ndim  # Ensure the axis is within the tensor dimensions
    shape = [1] * ndim  # Start with all dimensions as 1
    shape[axis] = tensor.size(0)  # Set the specified axis to the size of the tensor
    return tensor.view(*shape)


def blend(a: torch.Tensor, b: torch.Tensor, axis: int) -> torch.Tensor:
    """
    Blends two tensors `a` and `b` along the specified axis using linear interpolation.

    Args:
        a (torch.Tensor): The first tensor.
        b (torch.Tensor): The second tensor. Must have the same shape as `a`.
        axis (int): The axis along which to perform the blending.

    Returns:
        torch.Tensor: The blended tensor.
    """
    assert (
        a.shape == b.shape
    ), f"Tensors must have the same shape, got {a.shape} and {b.shape}"
    steps = a.size(axis)

    # Create a weight tensor that linearly interpolates from 0 to 1
    start = 1 / (steps + 1)
    end = steps / (steps + 1)
    weight = torch.linspace(start, end, steps=steps, device=a.device, dtype=a.dtype)

    # Make the weight tensor broadcastable across all dimensions
    weight = make_broadcastable(weight, axis, a.dim())

    # Perform the blending
    return a * (1 - weight) + b * weight


def blend_horizontal(a: torch.Tensor, b: torch.Tensor, overlap: int) -> torch.Tensor:
    if overlap == 0:
        return torch.cat([a, b], dim=-1)

    assert a.size(-1) >= overlap
    assert b.size(-1) >= overlap
    a_left, a_overlap = a[..., :-overlap], a[..., -overlap:]
    b_overlap, b_right = b[..., :overlap], b[..., overlap:]
    return torch.cat([a_left, blend(a_overlap, b_overlap, -1), b_right], dim=-1)


def blend_vertical(a: torch.Tensor, b: torch.Tensor, overlap: int) -> torch.Tensor:
    if overlap == 0:
        return torch.cat([a, b], dim=-2)

    assert a.size(-2) >= overlap
    assert b.size(-2) >= overlap
    a_top, a_overlap = a[..., :-overlap, :], a[..., -overlap:, :]
    b_overlap, b_bottom = b[..., :overlap, :], b[..., overlap:, :]
    return torch.cat([a_top, blend(a_overlap, b_overlap, -2), b_bottom], dim=-2)


def nearest_multiple(x: int, multiple: int) -> int:
    return round(x / multiple) * multiple


def apply_tiled(
    fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    num_tiles_w: int,
    num_tiles_h: int,
    overlap: int = 0,  # Number of pixel of overlap between adjacent tiles.
    # Use a factor of 2 times the latent downsample factor.
    min_block_size: int = 1,  # Minimum number of pixels in each dimension when subdividing.
):
    if num_tiles_w == 1 and num_tiles_h == 1:
        return fn(x)

    assert (
        num_tiles_w & (num_tiles_w - 1) == 0
    ), f"num_tiles_w={num_tiles_w} must be a power of 2"
    assert (
        num_tiles_h & (num_tiles_h - 1) == 0
    ), f"num_tiles_h={num_tiles_h} must be a power of 2"

    H, W = x.shape[-2:]
    assert H % min_block_size == 0
    assert W % min_block_size == 0
    ov = overlap // 2
    assert ov % min_block_size == 0

    if num_tiles_w >= 2:
        # Subdivide horizontally.
        half_W = nearest_multiple(W // 2, min_block_size)
        left = x[..., :, : half_W + ov]
        right = x[..., :, half_W - ov :]

        assert num_tiles_w % 2 == 0, f"num_tiles_w={num_tiles_w} must be even"
        left = apply_tiled(
            fn, left, num_tiles_w // 2, num_tiles_h, overlap, min_block_size
        )
        right = apply_tiled(
            fn, right, num_tiles_w // 2, num_tiles_h, overlap, min_block_size
        )
        if left is None or right is None:
            return None

        # If `fn` changed the resolution, adjust the overlap.
        resample_factor = left.size(-1) / (half_W + ov)
        out_overlap = int(overlap * resample_factor)

        return blend_horizontal(left, right, out_overlap)

    if num_tiles_h >= 2:
        # Subdivide vertically.
        half_H = nearest_multiple(H // 2, min_block_size)
        top = x[..., : half_H + ov, :]
        bottom = x[..., half_H - ov :, :]

        assert num_tiles_h % 2 == 0, f"num_tiles_h={num_tiles_h} must be even"
        top = apply_tiled(
            fn, top, num_tiles_w, num_tiles_h // 2, overlap, min_block_size
        )
        bottom = apply_tiled(
            fn, bottom, num_tiles_w, num_tiles_h // 2, overlap, min_block_size
        )
        if top is None or bottom is None:
            return None

        # If `fn` changed the resolution, adjust the overlap.
        resample_factor = top.size(-2) / (half_H + ov)
        out_overlap = int(overlap * resample_factor)

        return blend_vertical(top, bottom, out_overlap)

    raise ValueError(f"Invalid num_tiles_w={num_tiles_w} and num_tiles_h={num_tiles_h}")
