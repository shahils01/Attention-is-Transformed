import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adapter_trainer.accounting import build_replacement_summary, format_replacement_summary
from adapter_trainer.lgma_llama_attention import LlamaLgmaAttention, LlamaLgmaConfig


class FakeLlamaAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int = 32,
        num_heads: int = 4,
        num_key_value_heads: int = 4,
    ) -> None:
        super().__init__()
        head_dim = hidden_size // num_heads
        self.config = SimpleNamespace(
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            num_key_value_heads=num_key_value_heads,
            attention_dropout=0.0,
            attention_bias=False,
            rope_theta=10000.0,
            max_position_embeddings=64,
        )
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)


def test_llama_lgma_attention_preserves_output_shape():
    torch.manual_seed(0)
    module = LlamaLgmaAttention(
        LlamaLgmaConfig(
            hidden_size=32,
            num_attention_heads=4,
            qk_num_base_heads=2,
            value_num_base_heads=1,
            num_generators=2,
            attention_variant="lgma_residual",
        )
    )
    hidden_states = torch.randn(2, 8, 32)
    output, attn_weights, cache = module(hidden_states, output_attentions=True)
    assert output.shape == (2, 8, 32)
    assert attn_weights.shape == (2, 4, 8, 8)
    assert cache is None


def test_llama_lgma_attention_enforces_fixed_head_dims():
    module = LlamaLgmaAttention(
        LlamaLgmaConfig(
            hidden_size=48,
            num_attention_heads=6,
            qk_num_base_heads=3,
            value_num_base_heads=2,
            num_generators=2,
        )
    )
    assert module.num_heads == 6
    assert module.head_dim == 8
    assert module.base_dim == module.head_dim
    assert module.value_dim == module.head_dim


def test_separate_qk_and_value_base_heads_control_projection_widths():
    module = LlamaLgmaAttention(
        LlamaLgmaConfig(
            hidden_size=64,
            num_attention_heads=8,
            qk_num_base_heads=4,
            value_num_base_heads=2,
            num_generators=3,
        )
    )
    assert module.q_proj.weight.shape == (32, 64)
    assert module.k_proj.weight.shape == (32, 64)
    assert module.v_proj.weight.shape == (16, 64)
    assert module.qk_generated_heads_per_base == 2
    assert module.value_generated_heads_per_base == 4


def test_invalid_base_head_counts_raise():
    try:
        LlamaLgmaAttention(
            LlamaLgmaConfig(
                hidden_size=32,
                num_attention_heads=4,
                qk_num_base_heads=3,
                value_num_base_heads=1,
                num_generators=2,
            )
        )
    except ValueError as exc:
        assert "qk_num_base_heads" in str(exc)
    else:
        raise AssertionError("expected invalid qk_num_base_heads to raise")


def test_teacher_initialization_and_summary_printout_include_required_fields():
    torch.manual_seed(0)
    teacher = FakeLlamaAttention(hidden_size=32, num_heads=4, num_key_value_heads=2)
    student = LlamaLgmaAttention.from_teacher_attention(
        teacher,
        qk_num_base_heads=2,
        value_num_base_heads=1,
        num_generators=2,
        attention_variant="lgma_residual",
        layer_idx=0,
    )
    summary = build_replacement_summary(teacher, student, layer_idx=0)
    text = format_replacement_summary(summary)
    assert "teacher: hidden=32, heads=4, kv_heads=2, head_dim=8" in text
    assert "student: variant=lgma_residual, heads=4" in text
    assert "qk_base_heads=2" in text
    assert "value_base_heads=1" in text
    assert "q_width=16" in text
    assert "v_width=8" in text
    assert summary.student_num_heads == summary.teacher_num_heads
    assert summary.student_base_dim == summary.teacher_head_dim
    assert summary.student_value_dim == summary.teacher_head_dim


def test_adapter_state_round_trip(tmp_path):
    torch.manual_seed(0)
    module = LlamaLgmaAttention(
        LlamaLgmaConfig(
            hidden_size=32,
            num_attention_heads=4,
            qk_num_base_heads=2,
            value_num_base_heads=1,
            num_generators=2,
        )
    )
    path = tmp_path / "adapter.pt"
    torch.save({"config": module.config.__dict__, "state_dict": module.state_dict()}, path)
    from adapter_trainer.distill_tinyllama import load_adapter

    loaded = load_adapter(path, map_location="cpu")
    hidden_states = torch.randn(1, 4, 32)
    assert torch.allclose(module(hidden_states)[0], loaded(hidden_states)[0])
