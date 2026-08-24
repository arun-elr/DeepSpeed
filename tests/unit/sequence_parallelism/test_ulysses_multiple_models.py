# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch

from deepspeed import initialize
from deepspeed.accelerator import get_accelerator
from deepspeed.module_inject.tp_shard import get_num_kv_heads, get_shard_size_list, set_num_kv_heads
from deepspeed.sequence.layer import DistributedAttention, _SeqAllToAll
from deepspeed.utils import groups
from unit.common import DistributedTest
from unit.simple_model import SimpleModel


class _IdentityLocalAttn(torch.nn.Module):

    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads

    def forward(self, query, key, value, *args, **kwargs):
        return query


def _seq_batch_heads_hash(d0, d1, h, offset_d0=0, offset_d1=0, offset_h=0):
    d0 = d0 + offset_d0
    d1 = d1 + offset_d1
    h = h + offset_h
    return d0 * 10 + h + d1 * 0.1


class TestUlyssesMultipleModels(DistributedTest):
    world_size = 2

    def test_teacher_invertible_after_student(self):
        if get_accelerator().device_count() < 2:
            pytest.skip("This test requires 2 GPUs")

        set_num_kv_heads(99)
        autotp_before = get_num_kv_heads()

        hidden_dim = 10
        model = SimpleModel(hidden_dim)
        ds_engine, _, _, _ = initialize(
            model=model,
            config_params={
                "train_batch_size": 8,
                "data_parallel_size": 1,
                "sequence_parallel_size": 2
            },
        )
        spg = ds_engine.seq_parallel_group
        device = ds_engine.device
        rank = groups._get_sequence_parallel_rank()
        sp_size = groups._get_sequence_parallel_world_size()
        assert sp_size == 2

        teacher = DistributedAttention(_IdentityLocalAttn(3), spg, num_kv_heads=3)
        student = DistributedAttention(_IdentityLocalAttn(2), spg, num_kv_heads=2)
        assert teacher.total_heads == 3
        assert student.total_heads == 2

        local_seq, batch, head_dim = 4, 2, 16
        scatter_idx, gather_idx, batch_dim_idx = 2, 0, 1

        d0_idx = torch.arange(local_seq).reshape(-1, 1, 1, 1)
        d1_idx = torch.arange(batch).reshape(1, -1, 1, 1)
        teacher_h = torch.arange(3).reshape(1, 1, -1, 1)
        teacher_in = torch.zeros(local_seq, batch, 3, head_dim, device=device)
        teacher_in[:] = _seq_batch_heads_hash(d0_idx, d1_idx, teacher_h, local_seq * rank, 0)

        student_h = torch.arange(2).reshape(1, 1, -1, 1)
        student_in = torch.zeros(local_seq, batch, 2, head_dim, device=device)
        student_in[:] = _seq_batch_heads_hash(d0_idx, d1_idx, student_h, local_seq * rank, 0)

        def invert(tensor, total_heads):
            s2h = _SeqAllToAll.apply(spg, tensor, scatter_idx, gather_idx, batch_dim_idx, None, None, None, True,
                                     total_heads)
            h2s = _SeqAllToAll.apply(spg, s2h, gather_idx, scatter_idx, batch_dim_idx, None, None, None, True,
                                     total_heads)
            return s2h, h2s

        teacher_s2h, teacher_h2s = invert(teacher_in, teacher.total_heads)
        shard_list = get_shard_size_list(3, 2, num_kv_heads=3)
        assert teacher_s2h.shape[2] == shard_list[rank]
        assert torch.allclose(teacher_in, teacher_h2s)

        student_s2h, student_h2s = invert(student_in, student.total_heads)
        assert torch.allclose(student_in, student_h2s)
        assert student.total_heads == 2
        assert teacher.total_heads == 3

        teacher_s2h_again, teacher_h2s_again = invert(teacher_in, teacher.total_heads)
        assert teacher_s2h_again.shape[2] == shard_list[rank]
        assert torch.allclose(teacher_in, teacher_h2s_again)

        q = teacher_in.clone()
        k = teacher_in.clone()
        v = teacher_in.clone()
        teacher_out = teacher(q, k, v, batch_dim_idx)
        assert torch.allclose(teacher_out, teacher_in)

        assert teacher.total_heads == 3
        assert student.total_heads == 2
        assert get_num_kv_heads() == autotp_before
        assert get_num_kv_heads() == 99
