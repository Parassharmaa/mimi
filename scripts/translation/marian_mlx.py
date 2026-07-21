"""Minimal MLX implementation of the Marian architecture used by ElanMT.

The implementation deliberately mirrors Transformers 4.40.2. Full-prefix
greedy decoding remains the reference path. Incremental decoding supports the
shipping-shaped concatenating cache plus an opt-in block-growing self-K/V cache
for allocation experiments.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn


DIMENSIONS = 512
HEADS = 8
HEAD_DIMENSIONS = DIMENSIONS // HEADS
VOCABULARY_SIZE = 32_001
PAD_TOKEN_ID = 32_000
EOS_TOKEN_ID = 0
POSITION_TABLE_LENGTH = 192


ProjectedKVCache = tuple[mx.array, mx.array]


@dataclass(frozen=True)
class PackedLinearProjection:
    """Several output-aligned Linear projections executed as one matmul."""

    weight: mx.array
    scales: mx.array | None
    quantization_biases: mx.array | None
    output_bias: mx.array | None
    part_dimensions: tuple[int, ...]
    group_size: int | None
    bits: int | None
    mode: str | None

    def __call__(self, value: mx.array) -> mx.array:
        if self.scales is None:
            projected = value @ self.weight.T
        else:
            projected = mx.quantized_matmul(
                value,
                self.weight,
                scales=self.scales,
                biases=self.quantization_biases,
                transpose=True,
                group_size=self.group_size,
                bits=self.bits,
                mode=self.mode,
            )
        if self.output_bias is not None:
            projected = projected + self.output_bias
        return projected

    def split(self, value: mx.array) -> tuple[mx.array, ...]:
        parts = []
        start = 0
        for dimensions in self.part_dimensions:
            end = start + dimensions
            parts.append(value[..., start:end])
            start = end
        if start != value.shape[-1]:
            raise ValueError("packed projection output has an unexpected width")
        return tuple(parts)


def pack_linear_projections(
    projections: tuple[nn.Linear | nn.QuantizedLinear, ...],
) -> PackedLinearProjection:
    """Concatenate output rows without dequantizing or changing their order."""

    if len(projections) < 2:
        raise ValueError("packing requires at least two projections")
    quantized = all(isinstance(value, nn.QuantizedLinear) for value in projections)
    if not quantized and not all(isinstance(value, nn.Linear) for value in projections):
        raise ValueError("packed projections must share one dense or quantized type")
    input_dimensions = {int(value.weight.shape[-1]) for value in projections}
    if len(input_dimensions) != 1:
        raise ValueError("packed projections must share their input width")
    part_dimensions = tuple(int(value.bias.shape[0]) for value in projections)
    output_bias = mx.concatenate([value.bias for value in projections], axis=0)
    weight = mx.concatenate([value.weight for value in projections], axis=0)
    if quantized:
        group_sizes = {int(value.group_size) for value in projections}
        bit_widths = {int(value.bits) for value in projections}
        modes = {str(value.mode) for value in projections}
        if len(group_sizes) != 1 or len(bit_widths) != 1 or len(modes) != 1:
            raise ValueError("packed quantized projections use different contracts")
        scales = mx.concatenate([value.scales for value in projections], axis=0)
        quantization_biases = mx.concatenate(
            [value.biases for value in projections], axis=0
        )
        mx.eval(weight, scales, quantization_biases, output_bias)
        return PackedLinearProjection(
            weight=weight,
            scales=scales,
            quantization_biases=quantization_biases,
            output_bias=output_bias,
            part_dimensions=part_dimensions,
            group_size=next(iter(group_sizes)),
            bits=next(iter(bit_widths)),
            mode=next(iter(modes)),
        )
    mx.eval(weight, output_bias)
    return PackedLinearProjection(
        weight=weight,
        scales=None,
        quantization_biases=None,
        output_bias=output_bias,
        part_dimensions=part_dimensions,
        group_size=None,
        bits=None,
        mode=None,
    )


@dataclass(frozen=True)
class OutputProjectionShortlist:
    """Pre-sliced tied output projection for one source sentence."""

    token_ids: tuple[int, ...]
    weight: mx.array
    scales: mx.array | None
    biases: mx.array | None
    final_logits_bias: mx.array
    pad_index: int
    group_size: int | None
    bits: int | None
    mode: str | None


@dataclass(frozen=True)
class CompositeOutputProjection:
    """Static target-script projection plus a small source-specific extension."""

    parts: tuple[OutputProjectionShortlist, ...]
    token_ids: tuple[int, ...]
    pad_index: int


@dataclass(frozen=True)
class BlockGrowingKVCache:
    """Projected self-attention K/V storage grown in fixed-capacity blocks."""

    key: mx.array
    value: mx.array
    length: int
    block_size: int

    @property
    def capacity(self) -> int:
        return int(self.key.shape[2])

    @classmethod
    def from_update(
        cls,
        key: mx.array,
        value: mx.array,
        *,
        block_size: int,
    ) -> BlockGrowingKVCache:
        if block_size < 1:
            raise ValueError("self-attention cache block size must be positive")
        if key.shape != value.shape or key.ndim != 4 or key.shape[2] < 1:
            raise ValueError("self-attention K/V update has an invalid shape")
        update_length = int(key.shape[2])
        capacity = math.ceil(update_length / block_size) * block_size
        shape = (*key.shape[:2], capacity, key.shape[3])
        cache = cls(
            key=mx.zeros(shape, dtype=key.dtype),
            value=mx.zeros(shape, dtype=value.dtype),
            length=0,
            block_size=block_size,
        )
        return cache.append(key, value)

    def append(
        self,
        key: mx.array,
        value: mx.array,
    ) -> BlockGrowingKVCache:
        if (
            key.shape != value.shape
            or key.ndim != 4
            or key.shape[:2] != self.key.shape[:2]
            or key.shape[3] != self.key.shape[3]
        ):
            raise ValueError("self-attention K/V update is incompatible with its cache")
        update_length = int(key.shape[2])
        required = self.length + update_length
        next_key, next_value = self.key, self.value
        if required > self.capacity:
            next_capacity = math.ceil(required / self.block_size) * self.block_size
            extension_shape = (
                self.key.shape[0],
                self.key.shape[1],
                next_capacity - self.capacity,
                self.key.shape[3],
            )
            next_key = mx.concatenate(
                [next_key, mx.zeros(extension_shape, dtype=next_key.dtype)], axis=2
            )
            next_value = mx.concatenate(
                [next_value, mx.zeros(extension_shape, dtype=next_value.dtype)], axis=2
            )
        start = mx.array([self.length], dtype=mx.int32)
        next_key = mx.slice_update(next_key, key, start, axes=(2,))
        next_value = mx.slice_update(next_value, value, start, axes=(2,))
        return BlockGrowingKVCache(
            key=next_key,
            value=next_value,
            length=required,
            block_size=self.block_size,
        )

    def active(self) -> ProjectedKVCache:
        return (
            self.key[:, :, : self.length, :],
            self.value[:, :, : self.length, :],
        )


SelfKVCache = ProjectedKVCache | BlockGrowingKVCache
LayerKVCache = tuple[SelfKVCache, ProjectedKVCache]
_PRECOMPUTED_POSITION_TABLES: dict[str, mx.array] = {}


def positions(length: int, offset: int = 0) -> mx.array:
    indices = mx.arange(offset, offset + length, dtype=mx.float32)[:, None]
    inverse_frequency = mx.power(
        mx.array(10_000.0),
        -mx.arange(0, DIMENSIONS, 2, dtype=mx.float32) / DIMENSIONS,
    )[None, :]
    angles = indices * inverse_frequency
    return mx.concatenate([mx.sin(angles), mx.cos(angles)], axis=-1)


def precomputed_position_table(dtype: mx.Dtype) -> mx.array:
    """Return one evaluated 192x512 sinusoidal table per runtime/dtype."""

    key = str(dtype)
    table = _PRECOMPUTED_POSITION_TABLES.get(key)
    if table is None:
        table = positions(POSITION_TABLE_LENGTH).astype(dtype)
        mx.eval(table)
        _PRECOMPUTED_POSITION_TABLES[key] = table
    return table


def position_rows(
    length: int,
    offset: int,
    dtype: mx.Dtype,
    *,
    use_precomputed_table: bool,
) -> mx.array:
    if not use_precomputed_table:
        return positions(length, offset).astype(dtype)
    end = offset + length
    if length < 0 or offset < 0 or end > POSITION_TABLE_LENGTH:
        raise ValueError(
            "precomputed positional table supports positions "
            f"0..{POSITION_TABLE_LENGTH - 1}; requested offset={offset} length={length}"
        )
    return precomputed_position_table(dtype)[offset:end]


class Attention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.k_proj = nn.Linear(DIMENSIONS, DIMENSIONS, bias=True)
        self.v_proj = nn.Linear(DIMENSIONS, DIMENSIONS, bias=True)
        self.q_proj = nn.Linear(DIMENSIONS, DIMENSIONS, bias=True)
        self.out_proj = nn.Linear(DIMENSIONS, DIMENSIONS, bias=True)
        self.packed_qkv: PackedLinearProjection | None = None
        self.packed_kv: PackedLinearProjection | None = None

    def enable_qkv_packing(self) -> None:
        if self.packed_qkv is not None or self.packed_kv is not None:
            raise ValueError("attention projections are already packed")
        self.packed_qkv = pack_linear_projections(
            (self.q_proj, self.k_proj, self.v_proj)
        )
        self.q_proj = None
        self.k_proj = None
        self.v_proj = None

    def enable_kv_packing(self) -> None:
        if self.packed_qkv is not None or self.packed_kv is not None:
            raise ValueError("attention projections are already packed")
        self.packed_kv = pack_linear_projections((self.k_proj, self.v_proj))
        self.k_proj = None
        self.v_proj = None

    def project_qkv(self, value: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        if self.packed_qkv is None:
            if self.q_proj is None or self.k_proj is None or self.v_proj is None:
                raise ValueError("self-attention projection state is incomplete")
            return self.q_proj(value), self.k_proj(value), self.v_proj(value)
        query, key, projected_value = self.packed_qkv.split(self.packed_qkv(value))
        return query, key, projected_value

    def project_kv(self, value: mx.array) -> tuple[mx.array, mx.array]:
        if self.packed_kv is None:
            if self.k_proj is None or self.v_proj is None:
                raise ValueError("cross-attention projection state is incomplete")
            return self.k_proj(value), self.v_proj(value)
        key, projected_value = self.packed_kv.split(self.packed_kv(value))
        return key, projected_value

    @staticmethod
    def split_heads(value: mx.array) -> mx.array:
        batch, length, _ = value.shape
        return value.reshape(batch, length, HEADS, HEAD_DIMENSIONS).transpose(0, 2, 1, 3)

    @staticmethod
    def join_heads(value: mx.array) -> mx.array:
        batch, _, length, _ = value.shape
        return value.transpose(0, 2, 1, 3).reshape(batch, length, DIMENSIONS)

    def __call__(
        self,
        hidden_states: mx.array,
        key_value_states: mx.array | None = None,
        causal: bool = False,
    ) -> mx.array:
        if key_value_states is None:
            query, key, value = self.project_qkv(hidden_states)
        else:
            if self.q_proj is None:
                raise ValueError("cross-attention query projection is missing")
            query = self.q_proj(hidden_states)
            key, value = self.project_kv(key_value_states)
        query = self.split_heads(query)
        key = self.split_heads(key)
        value = self.split_heads(value)
        attended = mx.fast.scaled_dot_product_attention(
            query,
            key,
            value,
            scale=HEAD_DIMENSIONS**-0.5,
            mask="causal" if causal else None,
        )
        return self.out_proj(self.join_heads(attended))

    def prefill(
        self,
        hidden_states: mx.array,
        key_value_states: mx.array | None = None,
        causal: bool = False,
    ) -> tuple[mx.array, ProjectedKVCache]:
        """Run a parallel attention pass and retain its projected K/V state."""

        if key_value_states is None:
            query, key, value = self.project_qkv(hidden_states)
        else:
            if self.q_proj is None:
                raise ValueError("cross-attention query projection is missing")
            query = self.q_proj(hidden_states)
            key, value = self.project_kv(key_value_states)
        query = self.split_heads(query)
        key = self.split_heads(key)
        value = self.split_heads(value)
        attended = mx.fast.scaled_dot_product_attention(
            query,
            key,
            value,
            scale=HEAD_DIMENSIONS**-0.5,
            mask="causal" if causal else None,
        )
        return self.out_proj(self.join_heads(attended)), (key, value)

    def step(
        self,
        hidden_states: mx.array,
        *,
        key_value_states: mx.array | None = None,
        cache: SelfKVCache | None = None,
        self_cache_block_size: int | None = None,
    ) -> tuple[mx.array, SelfKVCache]:
        """Attend one decoder position and return reusable projected K/V state."""

        if self.packed_qkv is not None:
            if key_value_states is not None:
                raise ValueError("packed self-attention cannot project cross-attention")
            projected_query, projected_key, projected_value = self.project_qkv(
                hidden_states
            )
            query = self.split_heads(projected_query)
        else:
            if self.q_proj is None:
                raise ValueError("attention query projection is missing")
            query = self.split_heads(self.q_proj(hidden_states))
            projected_key = None
            projected_value = None
        if key_value_states is not None and cache is not None:
            if isinstance(cache, BlockGrowingKVCache):
                raise ValueError("cross-attention cannot use a growing self-K/V cache")
            key, value = cache
            next_cache = cache
        else:
            if projected_key is not None and projected_value is not None:
                next_key = self.split_heads(projected_key)
                next_value = self.split_heads(projected_value)
            else:
                source = hidden_states if key_value_states is None else key_value_states
                next_key_value, next_projected_value = self.project_kv(source)
                next_key = self.split_heads(next_key_value)
                next_value = self.split_heads(next_projected_value)
            if key_value_states is not None:
                key, value = next_key, next_value
                next_cache = (key, value)
            elif isinstance(cache, BlockGrowingKVCache):
                next_cache = cache.append(next_key, next_value)
                key, value = next_cache.active()
            elif cache is not None:
                key = mx.concatenate([cache[0], next_key], axis=2)
                value = mx.concatenate([cache[1], next_value], axis=2)
                next_cache = (key, value)
            elif self_cache_block_size is not None:
                next_cache = BlockGrowingKVCache.from_update(
                    next_key,
                    next_value,
                    block_size=self_cache_block_size,
                )
                key, value = next_cache.active()
            else:
                key, value = next_key, next_value
                next_cache = (key, value)
        attended = mx.fast.scaled_dot_product_attention(
            query,
            key,
            value,
            scale=HEAD_DIMENSIONS**-0.5,
        )
        return self.out_proj(self.join_heads(attended)), next_cache


class EncoderLayer(nn.Module):
    def __init__(self, ffn_dimensions: int = 2_048) -> None:
        super().__init__()
        self.self_attn = Attention()
        self.self_attn_layer_norm = nn.LayerNorm(DIMENSIONS)
        self.fc1 = nn.Linear(DIMENSIONS, ffn_dimensions, bias=True)
        self.fc2 = nn.Linear(ffn_dimensions, DIMENSIONS, bias=True)
        self.final_layer_norm = nn.LayerNorm(DIMENSIONS)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden_states = self.self_attn_layer_norm(
            hidden_states + self.self_attn(hidden_states)
        )
        feed_forward = self.fc2(nn.silu(self.fc1(hidden_states)))
        return self.final_layer_norm(hidden_states + feed_forward)


class DecoderLayer(nn.Module):
    def __init__(self, ffn_dimensions: int = 2_048) -> None:
        super().__init__()
        self.self_attn = Attention()
        self.self_attn_layer_norm = nn.LayerNorm(DIMENSIONS)
        self.encoder_attn = Attention()
        self.encoder_attn_layer_norm = nn.LayerNorm(DIMENSIONS)
        self.fc1 = nn.Linear(DIMENSIONS, ffn_dimensions, bias=True)
        self.fc2 = nn.Linear(ffn_dimensions, DIMENSIONS, bias=True)
        self.final_layer_norm = nn.LayerNorm(DIMENSIONS)

    def __call__(self, hidden_states: mx.array, encoder_states: mx.array) -> mx.array:
        hidden_states = self.self_attn_layer_norm(
            hidden_states + self.self_attn(hidden_states, causal=True)
        )
        hidden_states = self.encoder_attn_layer_norm(
            hidden_states + self.encoder_attn(hidden_states, encoder_states)
        )
        feed_forward = self.fc2(nn.silu(self.fc1(hidden_states)))
        return self.final_layer_norm(hidden_states + feed_forward)

    def prefill(
        self,
        hidden_states: mx.array,
        encoder_states: mx.array,
    ) -> tuple[mx.array, LayerKVCache]:
        """Run a causal teacher-forced pass and return reusable decoder caches."""

        attended, self_cache = self.self_attn.prefill(hidden_states, causal=True)
        hidden_states = self.self_attn_layer_norm(hidden_states + attended)
        attended, cross_cache = self.encoder_attn.prefill(
            hidden_states,
            key_value_states=encoder_states,
        )
        hidden_states = self.encoder_attn_layer_norm(hidden_states + attended)
        feed_forward = self.fc2(nn.silu(self.fc1(hidden_states)))
        hidden_states = self.final_layer_norm(hidden_states + feed_forward)
        return hidden_states, (self_cache, cross_cache)

    def step(
        self,
        hidden_states: mx.array,
        encoder_states: mx.array,
        cache: LayerKVCache | None = None,
        self_cache_block_size: int | None = None,
    ) -> tuple[mx.array, LayerKVCache]:
        self_cache = cache[0] if cache is not None else None
        cross_cache = cache[1] if cache is not None else None
        attended, next_self_cache = self.self_attn.step(
            hidden_states,
            cache=self_cache,
            self_cache_block_size=self_cache_block_size,
        )
        hidden_states = self.self_attn_layer_norm(hidden_states + attended)
        attended, next_cross_cache = self.encoder_attn.step(
            hidden_states,
            key_value_states=encoder_states,
            cache=cross_cache,
        )
        hidden_states = self.encoder_attn_layer_norm(hidden_states + attended)
        feed_forward = self.fc2(nn.silu(self.fc1(hidden_states)))
        hidden_states = self.final_layer_norm(hidden_states + feed_forward)
        return hidden_states, (next_self_cache, next_cross_cache)


class Encoder(nn.Module):
    def __init__(self, layer_count: int = 6, ffn_dimensions: int = 2_048) -> None:
        super().__init__()
        if layer_count < 1:
            raise ValueError("Marian encoder must contain at least one layer")
        self.layers = [EncoderLayer(ffn_dimensions) for _ in range(layer_count)]

    def __call__(self, hidden_states: mx.array) -> mx.array:
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class Decoder(nn.Module):
    def __init__(self, layer_count: int = 6, ffn_dimensions: int = 2_048) -> None:
        super().__init__()
        if layer_count < 1:
            raise ValueError("Marian decoder must contain at least one layer")
        self.layers = [DecoderLayer(ffn_dimensions) for _ in range(layer_count)]

    def __call__(self, hidden_states: mx.array, encoder_states: mx.array) -> mx.array:
        for layer in self.layers:
            hidden_states = layer(hidden_states, encoder_states)
        return hidden_states

    def prefill(
        self,
        hidden_states: mx.array,
        encoder_states: mx.array,
    ) -> tuple[mx.array, list[LayerKVCache]]:
        next_caches = []
        for layer in self.layers:
            hidden_states, cache = layer.prefill(hidden_states, encoder_states)
            next_caches.append(cache)
        return hidden_states, next_caches

    def step(
        self,
        hidden_states: mx.array,
        encoder_states: mx.array,
        caches: list[LayerKVCache] | None = None,
        self_cache_block_size: int | None = None,
    ) -> tuple[mx.array, list[LayerKVCache]]:
        next_caches = []
        for index, layer in enumerate(self.layers):
            cache = caches[index] if caches is not None else None
            hidden_states, next_cache = layer.step(
                hidden_states,
                encoder_states,
                cache,
                self_cache_block_size,
            )
            next_caches.append(next_cache)
        return hidden_states, next_caches


class Marian(nn.Module):
    def __init__(
        self,
        *,
        encoder_layers: int = 6,
        decoder_layers: int = 6,
        encoder_ffn_dimensions: int = 2_048,
        decoder_ffn_dimensions: int = 2_048,
    ) -> None:
        super().__init__()
        self.shared = nn.Embedding(VOCABULARY_SIZE, DIMENSIONS)
        self.encoder = Encoder(encoder_layers, encoder_ffn_dimensions)
        self.decoder = Decoder(decoder_layers, decoder_ffn_dimensions)
        self.final_logits_bias = mx.zeros((1, VOCABULARY_SIZE))
        self.use_precomputed_position_table = False
        self.packed_attention_projections = False

    def enable_packed_attention_projections(self) -> None:
        if self.packed_attention_projections:
            raise ValueError("attention projections are already packed")
        for layer in self.encoder.layers:
            layer.self_attn.enable_qkv_packing()
        for layer in self.decoder.layers:
            layer.self_attn.enable_qkv_packing()
            layer.encoder_attn.enable_kv_packing()
        mx.synchronize()
        self.packed_attention_projections = True

    def enable_precomputed_position_table(self) -> None:
        # Prime the runtime cache during model preparation rather than charging
        # the first benchmarked encoder invocation for table construction.
        precomputed_position_table(self.final_logits_bias.dtype)
        self.use_precomputed_position_table = True

    def encode(self, input_ids: mx.array) -> mx.array:
        length = input_ids.shape[1]
        hidden_states = self.shared(input_ids) * math.sqrt(DIMENSIONS)
        hidden_states = hidden_states + position_rows(
            length,
            0,
            hidden_states.dtype,
            use_precomputed_table=self.use_precomputed_position_table,
        )
        return self.encoder(hidden_states)

    def prepare_output_shortlist(
        self, token_ids: list[int] | tuple[int, ...]
    ) -> OutputProjectionShortlist:
        ordered = tuple(int(value) for value in token_ids)
        if (
            not ordered
            or tuple(sorted(set(ordered))) != ordered
            or ordered[0] < 0
            or ordered[-1] >= VOCABULARY_SIZE
            or EOS_TOKEN_ID not in ordered
            or PAD_TOKEN_ID not in ordered
        ):
            raise ValueError(
                "output shortlist must be sorted, unique, in-vocabulary, and include EOS/PAD"
            )
        return self._prepare_output_projection(ordered, ordered.index(PAD_TOKEN_ID))

    def prepare_output_extension(
        self, token_ids: list[int] | tuple[int, ...]
    ) -> OutputProjectionShortlist:
        ordered = tuple(int(value) for value in token_ids)
        if (
            not ordered
            or tuple(sorted(set(ordered))) != ordered
            or ordered[0] < 0
            or ordered[-1] >= VOCABULARY_SIZE
            or EOS_TOKEN_ID in ordered
            or PAD_TOKEN_ID in ordered
        ):
            raise ValueError(
                "output extension must be sorted, unique, in-vocabulary, and exclude EOS/PAD"
            )
        return self._prepare_output_projection(ordered, -1)

    def _prepare_output_projection(
        self,
        ordered: tuple[int, ...],
        pad_index: int,
    ) -> OutputProjectionShortlist:
        indices = mx.array(ordered, dtype=mx.int32)
        weight = self.shared.weight[indices]
        final_bias = self.final_logits_bias[:, indices]
        if isinstance(self.shared, nn.QuantizedEmbedding):
            scales = self.shared.scales[indices]
            source_biases = self.shared.get("biases")
            biases = source_biases[indices] if source_biases is not None else None
            values = [weight, scales, final_bias]
            if biases is not None:
                values.append(biases)
            mx.eval(*values)
            return OutputProjectionShortlist(
                token_ids=ordered,
                weight=weight,
                scales=scales,
                biases=biases,
                final_logits_bias=final_bias,
                pad_index=pad_index,
                group_size=self.shared.group_size,
                bits=self.shared.bits,
                mode=self.shared.mode,
            )
        mx.eval(weight, final_bias)
        return OutputProjectionShortlist(
            token_ids=ordered,
            weight=weight,
            scales=None,
            biases=None,
            final_logits_bias=final_bias,
            pad_index=pad_index,
            group_size=None,
            bits=None,
            mode=None,
        )

    def project_output(
        self,
        hidden_states: mx.array,
        shortlist: OutputProjectionShortlist | CompositeOutputProjection | None = None,
    ) -> mx.array:
        if shortlist is None:
            return self.shared.as_linear(hidden_states) + self.final_logits_bias
        if isinstance(shortlist, CompositeOutputProjection):
            return mx.concatenate(
                [self.project_output(hidden_states, part) for part in shortlist.parts],
                axis=-1,
            )
        if shortlist.scales is not None:
            logits = mx.quantized_matmul(
                hidden_states,
                shortlist.weight,
                scales=shortlist.scales,
                biases=shortlist.biases,
                transpose=True,
                group_size=shortlist.group_size,
                bits=shortlist.bits,
                mode=shortlist.mode,
            )
        else:
            logits = hidden_states @ shortlist.weight.T
        return logits + shortlist.final_logits_bias

    def decode(self, decoder_ids: mx.array, encoder_states: mx.array) -> mx.array:
        length = decoder_ids.shape[1]
        hidden_states = self.shared(decoder_ids) * math.sqrt(DIMENSIONS)
        hidden_states = hidden_states + positions(length).astype(hidden_states.dtype)
        hidden_states = self.decoder(hidden_states, encoder_states)
        return self.project_output(hidden_states)

    def decode_prefill(
        self,
        decoder_ids: mx.array,
        encoder_states: mx.array,
    ) -> tuple[mx.array, list[LayerKVCache]]:
        """Decode a causal prefix in parallel and retain caches for continuation."""

        length = decoder_ids.shape[1]
        hidden_states = self.shared(decoder_ids) * math.sqrt(DIMENSIONS)
        hidden_states = hidden_states + position_rows(
            length,
            0,
            hidden_states.dtype,
            use_precomputed_table=self.use_precomputed_position_table,
        )
        hidden_states, caches = self.decoder.prefill(hidden_states, encoder_states)
        logits = self.project_output(hidden_states)
        return logits, caches

    def decode_step(
        self,
        decoder_id: int,
        encoder_states: mx.array,
        caches: list[LayerKVCache] | None,
        position_offset: int,
        self_cache_block_size: int | None = None,
        output_shortlist: OutputProjectionShortlist | CompositeOutputProjection | None = None,
    ) -> tuple[mx.array, list[LayerKVCache]]:
        decoder_ids = mx.array([[decoder_id]], dtype=mx.int32)
        hidden_states = self.shared(decoder_ids) * math.sqrt(DIMENSIONS)
        hidden_states = hidden_states + position_rows(
            1,
            position_offset,
            hidden_states.dtype,
            use_precomputed_table=self.use_precomputed_position_table,
        )
        hidden_states, next_caches = self.decoder.step(
            hidden_states,
            encoder_states,
            caches,
            self_cache_block_size,
        )
        logits = self.project_output(hidden_states, output_shortlist)
        return logits, next_caches

    def generate(self, input_ids: list[int], maximum_tokens: int = 192) -> list[int]:
        encoder_states = self.encode(mx.array([input_ids], dtype=mx.int32))
        decoder_ids = [PAD_TOKEN_ID]
        output: list[int] = []
        for _ in range(maximum_tokens):
            logits = self.decode(
                mx.array([decoder_ids], dtype=mx.int32), encoder_states
            )[0, -1]
            logits[PAD_TOKEN_ID] = -1e9
            token = int(mx.argmax(logits).item())
            if token == EOS_TOKEN_ID:
                break
            output.append(token)
            decoder_ids.append(token)
        return output

    def generate_cached(
        self,
        input_ids: list[int],
        maximum_tokens: int = 192,
        *,
        self_cache_block_size: int | None = None,
    ) -> list[int]:
        """Greedily decode with incremental self- and cross-attention K/V caches.

        When ``self_cache_block_size`` is omitted, the established exact path
        concatenates one projected self-attention position per step. A positive
        block size opts into capacity-rounded storage; cross-attention projections
        remain immutable after their first decoder step in both modes.
        """

        if self_cache_block_size is not None and self_cache_block_size < 1:
            raise ValueError("self-attention cache block size must be positive")

        encoder_states = self.encode(mx.array([input_ids], dtype=mx.int32))
        return self._generate_cached_from_encoder(
            encoder_states,
            maximum_tokens,
            self_cache_block_size=self_cache_block_size,
        )

    def generate_cached_shortlist(
        self,
        input_ids: list[int],
        output_token_ids: list[int] | tuple[int, ...],
        maximum_tokens: int = 192,
    ) -> list[int]:
        shortlist = self.prepare_output_shortlist(output_token_ids)
        encoder_states = self.encode(mx.array([input_ids], dtype=mx.int32))
        return self._generate_cached_from_encoder(
            encoder_states,
            maximum_tokens,
            output_shortlist=shortlist,
        )

    def generate_cached_prepared_shortlist(
        self,
        input_ids: list[int],
        output_shortlist: OutputProjectionShortlist | CompositeOutputProjection,
        maximum_tokens: int = 192,
    ) -> list[int]:
        encoder_states = self.encode(mx.array([input_ids], dtype=mx.int32))
        return self._generate_cached_from_encoder(
            encoder_states,
            maximum_tokens,
            output_shortlist=output_shortlist,
        )

    def _generate_cached_from_encoder(
        self,
        encoder_states: mx.array,
        maximum_tokens: int,
        *,
        self_cache_block_size: int | None = None,
        output_shortlist: OutputProjectionShortlist | CompositeOutputProjection | None = None,
    ) -> list[int]:
        decoder_id = PAD_TOKEN_ID
        caches = None
        output: list[int] = []
        for position_offset in range(maximum_tokens):
            logits, caches = self.decode_step(
                decoder_id,
                encoder_states,
                caches,
                position_offset,
                self_cache_block_size,
                output_shortlist,
            )
            next_logits = logits[0, -1]
            pad_index = (
                PAD_TOKEN_ID if output_shortlist is None else output_shortlist.pad_index
            )
            next_logits[pad_index] = -1e9
            local_token = int(mx.argmax(next_logits).item())
            token = (
                local_token
                if output_shortlist is None
                else output_shortlist.token_ids[local_token]
            )
            if token == EOS_TOKEN_ID:
                break
            output.append(token)
            decoder_id = token
        return output

    def generate_beam(
        self,
        input_ids: list[int],
        beam_size: int = 4,
        maximum_tokens: int = 192,
        length_penalty: float = 1.0,
    ) -> list[int]:
        """Decode with deterministic length-normalized beam search.

        ElanMT's pinned generation configuration specifies four beams. This
        implementation keeps the decoder deliberately simple and auditable;
        it does not sample and never admits the pad token as an output.
        """

        if beam_size < 2:
            return self.generate(input_ids, maximum_tokens)
        candidates = self.generate_beam_nbest(
            input_ids,
            beam_size=beam_size,
            maximum_tokens=maximum_tokens,
            length_penalty=length_penalty,
            num_return_sequences=1,
        )
        return candidates[0][0] if candidates else []

    def generate_beam_nbest(
        self,
        input_ids: list[int],
        beam_size: int = 4,
        maximum_tokens: int = 192,
        length_penalty: float = 1.0,
        num_return_sequences: int = 4,
    ) -> list[tuple[list[int], float]]:
        """Return deterministic length-normalized beam candidates and scores."""

        if beam_size < 2:
            raise ValueError("n-best beam size must be at least two")
        if num_return_sequences < 1 or num_return_sequences > beam_size:
            raise ValueError("n-best return count must be in 1...beam_size")
        if length_penalty <= 0:
            raise ValueError("length_penalty must be positive")

        encoder_states = self.encode(mx.array([input_ids], dtype=mx.int32))
        active: list[tuple[list[int], float]] = [([PAD_TOKEN_ID], 0.0)]
        finished: list[tuple[list[int], float]] = []

        def normalized_score(generated_length: int, score: float) -> float:
            return score / (max(1, generated_length) ** length_penalty)

        for _ in range(maximum_tokens):
            candidates: list[tuple[list[int], float, int]] = []
            for decoder_ids, cumulative_score in active:
                logits = self.decode(
                    mx.array([decoder_ids], dtype=mx.int32), encoder_states
                )[0, -1]
                logits[PAD_TOKEN_ID] = -1e9
                log_probabilities = logits - mx.logsumexp(logits)
                token_ids = mx.argpartition(
                    log_probabilities, -(2 * beam_size)
                )[-(2 * beam_size):]
                token_scores = log_probabilities[token_ids]
                mx.eval(token_ids, token_scores)
                for token, token_score in zip(
                    token_ids.tolist(), token_scores.tolist(), strict=True
                ):
                    score = cumulative_score + float(token_score)
                    candidates.append((decoder_ids, score, int(token)))

            candidates.sort(key=lambda item: item[1], reverse=True)
            next_active: list[tuple[list[int], float]] = []
            for rank, (decoder_ids, score, token) in enumerate(candidates):
                if token == EOS_TOKEN_ID:
                    if rank < beam_size:
                        finished.append(
                            (
                                decoder_ids[1:],
                                normalized_score(len(decoder_ids), score),
                            )
                        )
                        finished.sort(key=lambda item: item[1], reverse=True)
                        del finished[beam_size:]
                    continue
                next_active.append((decoder_ids + [token], score))
                if len(next_active) == beam_size:
                    break

            if not next_active:
                break
            active = next_active
            if len(finished) == beam_size:
                generated_length = len(active[0][0]) - 1
                highest_attainable = normalized_score(generated_length, active[0][1])
                if finished[-1][1] >= highest_attainable:
                    break

        finished.extend(
            (tokens[1:], normalized_score(len(tokens) - 1, score))
            for tokens, score in active
        )
        if not finished:
            return []
        ranked: list[tuple[list[int], float]] = []
        seen: set[tuple[int, ...]] = set()
        for tokens, score in sorted(finished, key=lambda item: item[1], reverse=True):
            identity = tuple(tokens)
            if identity in seen:
                continue
            seen.add(identity)
            ranked.append((tokens, score))
            if len(ranked) == num_return_sequences:
                break
        return ranked

    def generate_with_diagnostics(
        self, input_ids: list[int], maximum_tokens: int = 192
    ) -> tuple[list[int], dict]:
        """Greedily decode and expose auditable student-only selection features."""

        encoder_states = self.encode(mx.array([input_ids], dtype=mx.int32))
        pooled = mx.mean(encoder_states[0].astype(mx.float32), axis=0)
        pooled_norm = mx.sqrt(mx.sum(pooled * pooled))
        pooled = pooled / mx.maximum(pooled_norm, mx.array(1e-12))
        decoder_ids = [PAD_TOKEN_ID]
        output: list[int] = []
        token_nll: list[float] = []
        for _ in range(maximum_tokens):
            logits = self.decode(
                mx.array([decoder_ids], dtype=mx.int32), encoder_states
            )[0, -1]
            logits[PAD_TOKEN_ID] = -1e9
            token = int(mx.argmax(logits).item())
            token_nll.append(float((mx.logsumexp(logits) - logits[token]).item()))
            if token == EOS_TOKEN_ID:
                break
            output.append(token)
            decoder_ids.append(token)
        mx.synchronize()
        return output, {
            "student_sequence_nll": sum(token_nll) / max(1, len(token_nll)),
            "encoder_embedding": pooled.tolist(),
        }


def infer_layer_count(weight_names: list[str], stack: str) -> int:
    prefix = f"{stack}.layers."
    indices = {
        int(name.removeprefix(prefix).split(".", 1)[0])
        for name in weight_names
        if name.startswith(prefix)
    }
    if not indices:
        raise ValueError(f"Marian weights contain no {stack} layers")
    expected = set(range(max(indices) + 1))
    if indices != expected:
        raise ValueError(
            f"Marian {stack} layer indices must be contiguous: {sorted(indices)}"
        )
    return len(indices)


def infer_ffn_dimensions(weights: dict[str, mx.array], stack: str) -> int:
    prefix = f"{stack}.layers.0.fc1."
    for suffix in ("scales", "weight"):
        value = weights.get(f"{prefix}{suffix}")
        if value is not None:
            dimensions = int(value.shape[0])
            if dimensions < 1:
                break
            return dimensions
    raise ValueError(f"Marian weights do not declare the {stack} FFN dimensions")


def load_model(
    weights_path: Path,
    *,
    quantization_bits: int | None = None,
    quantization_group_size: int = 64,
    precompute_position_table: bool = False,
    pack_attention_projections: bool = False,
) -> Marian:
    weights = mx.load(str(weights_path))
    renamed = []
    for name, value in weights.items():
        if name.startswith("model."):
            name = name.removeprefix("model.")
        renamed.append((name, value))
    renamed_weights = dict(renamed)
    names = list(renamed_weights)
    model = Marian(
        encoder_layers=infer_layer_count(names, "encoder"),
        decoder_layers=infer_layer_count(names, "decoder"),
        encoder_ffn_dimensions=infer_ffn_dimensions(renamed_weights, "encoder"),
        decoder_ffn_dimensions=infer_ffn_dimensions(renamed_weights, "decoder"),
    )
    if quantization_bits is not None:
        nn.quantize(
            model,
            group_size=quantization_group_size,
            bits=quantization_bits,
        )
    model.load_weights(renamed, strict=True)
    mx.eval(model.parameters())
    if pack_attention_projections:
        model.enable_packed_attention_projections()
    if precompute_position_table:
        model.enable_precomputed_position_table()
    return model
