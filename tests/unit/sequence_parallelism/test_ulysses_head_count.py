# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch
from unittest.mock import patch

from deepspeed.module_inject import tp_shard
from deepspeed.sequence import layer as ulysses_layer
from deepspeed.sequence.layer import (DistributedAttention, _UNSET, _capture_total_heads_from_scatter,
                                      _probe_local_attn_num_heads, _resolve_total_heads, get_ulysses_num_kv_heads,
                                      set_ulysses_num_kv_heads)


class _DummyAttn(torch.nn.Module):

    def __init__(self, **attrs):
        super().__init__()
        for name, value in attrs.items():
            setattr(self, name, value)

    def forward(self, query, key, value, *args, **kwargs):
        return query


class _DummyConfig:

    def __init__(self, **attrs):
        for name, value in attrs.items():
            setattr(self, name, value)


@pytest.fixture(autouse=True)
def _reset_head_globals():
    ulysses_layer._ulysses_num_kv_heads = None
    tp_shard.num_kv_heads = None
    yield
    ulysses_layer._ulysses_num_kv_heads = None
    tp_shard.num_kv_heads = None


def test_unset_is_not_none():
    assert _UNSET is not None
    assert _UNSET is not False
    assert _UNSET is ulysses_layer._UNSET


@pytest.mark.parametrize("attr", [
    "num_key_value_heads",
    "num_kv_heads",
    "num_attention_heads",
    "num_heads",
    "n_heads",
    "n_head",
])
def test_probe_local_attn_attribute(attr):
    attn = _DummyAttn(**{attr: 5})
    assert _probe_local_attn_num_heads(attn) == 5


@pytest.mark.parametrize("attr", [
    "num_key_value_heads",
    "num_kv_heads",
    "num_attention_heads",
    "num_heads",
    "n_heads",
    "n_head",
])
def test_probe_local_attn_config_attribute(attr):
    attn = _DummyAttn(config=_DummyConfig(**{attr: 6}))
    assert _probe_local_attn_num_heads(attn) == 6


def test_probe_prefers_module_over_config():
    attn = _DummyAttn(num_heads=3, config=_DummyConfig(num_heads=9))
    assert _probe_local_attn_num_heads(attn) == 3


def test_probe_returns_none_when_missing():
    assert _probe_local_attn_num_heads(_DummyAttn()) is None
    assert _probe_local_attn_num_heads(None) is None


def test_ctor_records_explicit_num_kv_heads():
    attn = DistributedAttention(_DummyAttn(num_heads=8), None, num_kv_heads=3)
    assert attn.total_heads == 3


def test_ctor_records_probed_heads_when_arg_omitted():
    attn = DistributedAttention(_DummyAttn(num_key_value_heads=4), None)
    assert attn.total_heads == 4


def test_ctor_explicit_none_does_not_read_shim():
    ulysses_layer._ulysses_num_kv_heads = 7
    attn = DistributedAttention(_DummyAttn(), None, num_kv_heads=None)
    assert attn.total_heads is None
    assert get_ulysses_num_kv_heads() == 7


def test_capture_only_on_uneven_scatter():
    even = torch.zeros(2, 4, 4, 8)
    uneven = torch.zeros(2, 4, 3, 8)
    gather_dir = torch.zeros(2, 4, 1, 8)

    assert _capture_total_heads_from_scatter(even, scatter_idx=2, seq_world_size=2) is None
    assert _capture_total_heads_from_scatter(uneven, scatter_idx=0, seq_world_size=2) is None
    assert _capture_total_heads_from_scatter(uneven, scatter_idx=1, seq_world_size=2) is None
    assert _capture_total_heads_from_scatter(gather_dir, scatter_idx=0, seq_world_size=2) is None
    assert _capture_total_heads_from_scatter(uneven, scatter_idx=2, seq_world_size=2) == 3

    too_small = torch.zeros(2, 4, 1, 8)
    with pytest.raises(AssertionError, match="must be larger than sequence parallel size"):
        _capture_total_heads_from_scatter(too_small, scatter_idx=2, seq_world_size=2)


def test_explicit_none_does_not_read_shim():
    ulysses_layer._ulysses_num_kv_heads = 7
    even = torch.zeros(2, 4, 4, 8)
    with patch.object(ulysses_layer.dist, "get_world_size", return_value=2):
        resolved = _resolve_total_heads(even, scatter_idx=2, group=None, total_heads=None, allow_global_fallback=False)
    assert resolved is None
    assert get_ulysses_num_kv_heads() == 7


def test_unset_with_fallback_reads_shim():
    ulysses_layer._ulysses_num_kv_heads = 7
    even = torch.zeros(2, 4, 4, 8)
    with patch.object(ulysses_layer.dist, "get_world_size", return_value=2):
        resolved = _resolve_total_heads(even,
                                        scatter_idx=2,
                                        group=None,
                                        total_heads=_UNSET,
                                        allow_global_fallback=True)
    assert resolved == 7


def test_resolve_does_not_write_autotp_global():
    tp_shard.set_num_kv_heads(99)
    uneven = torch.zeros(2, 4, 3, 8)
    with patch.object(ulysses_layer.dist, "get_world_size", return_value=2):
        resolved = _resolve_total_heads(uneven,
                                        scatter_idx=2,
                                        group=None,
                                        total_heads=_UNSET,
                                        allow_global_fallback=True)
    assert resolved == 3
    assert tp_shard.get_num_kv_heads() == 99
    assert get_ulysses_num_kv_heads() == 3


def test_resolve_explicit_none_does_not_write_shim():
    uneven = torch.zeros(2, 4, 3, 8)
    with patch.object(ulysses_layer.dist, "get_world_size", return_value=2):
        resolved = _resolve_total_heads(uneven,
                                        scatter_idx=2,
                                        group=None,
                                        total_heads=None,
                                        allow_global_fallback=False)
    assert resolved == 3
    assert get_ulysses_num_kv_heads() is None
    assert tp_shard.get_num_kv_heads() is None


def test_two_instances_isolated():
    teacher = DistributedAttention(_DummyAttn(num_heads=3), None, num_kv_heads=3)
    student = DistributedAttention(_DummyAttn(num_heads=5), None, num_kv_heads=5)
    assert teacher.total_heads == 3
    assert student.total_heads == 5

    lazy_a = DistributedAttention(_DummyAttn(), None)
    lazy_b = DistributedAttention(_DummyAttn(), None)
    assert lazy_a.total_heads is None
    assert lazy_b.total_heads is None

    with patch.object(ulysses_layer.dist, "get_world_size", return_value=2):
        lazy_a._bind_total_heads(torch.zeros(2, 4, 3, 8))
        lazy_b._bind_total_heads(torch.zeros(2, 4, 5, 8))
        teacher._bind_total_heads(torch.zeros(2, 4, 7, 8))

    assert lazy_a.total_heads == 3
    assert lazy_b.total_heads == 5
    assert teacher.total_heads == 3
    assert student.total_heads == 5
    assert get_ulysses_num_kv_heads() is None
    assert tp_shard.get_num_kv_heads() is None


def test_bind_skips_even_scatter():
    attn = DistributedAttention(_DummyAttn(), None)
    with patch.object(ulysses_layer.dist, "get_world_size", return_value=2):
        attn._bind_total_heads(torch.zeros(2, 4, 4, 8))
    assert attn.total_heads is None


def test_set_ulysses_num_kv_heads_warns():
    with pytest.warns(DeprecationWarning, match="set_ulysses_num_kv_heads"):
        set_ulysses_num_kv_heads(4)
    assert get_ulysses_num_kv_heads() == 4
    assert tp_shard.get_num_kv_heads() is None
