import numpy as np
import torch
import torch.nn.functional as F

from lgma.attention import LieGeneratedMetricAttention
from lgma.baselines import GroupedQueryAttention
from lgma.seq2seq import TokenBucketBatcher, WMT14Transformer, collate_translation


def test_truncation_preserves_eos_for_source_and_target():
    batch = collate_translation(
        [(np.array([5, 6, 7, 8, 2]), np.array([9, 10, 11, 12, 2]))],
        pad_id=3,
        bos_id=1,
        eos_id=2,
        max_source_length=4,
        max_target_length=4,
    )

    assert batch.source_ids.tolist() == [[5, 6, 7, 2]]
    assert batch.target_ids.tolist() == [[9, 10, 11, 2]]
    assert batch.decoder_input_ids.tolist() == [[1, 9, 10, 11]]


def test_tied_embedding_initialization_has_transformer_scale_and_sane_loss():
    torch.manual_seed(7)
    model = WMT14Transformer(
        vocab_size=256,
        d_model=32,
        ffn_dim=64,
        num_encoder_layers=1,
        num_decoder_layers=1,
        num_heads=4,
        head_dim=8,
        dropout=0.0,
        max_source_length=8,
        max_target_length=8,
        share_all_embeddings=True,
    )

    assert model.output_projection.weight is model.source_embedding.weight
    assert torch.count_nonzero(model.source_embedding.weight[model.pad_id]) == 0
    assert 0.12 < model.source_embedding.weight[model.source_embedding.weight.ne(0)].std().item() < 0.23

    source = torch.tensor([[4, 5, 2], [6, 7, 2]])
    decoder_input = torch.tensor([[1, 8, 9], [1, 10, 11]])
    targets = torch.tensor([[8, 9, 2], [10, 11, 2]])
    logits = model(source, decoder_input)
    loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())

    assert torch.isfinite(loss)
    assert loss.item() < 15.0


def test_wmt_replaces_decoder_cross_attention_with_selected_variant():
    common = dict(
        vocab_size=256,
        d_model=32,
        ffn_dim=64,
        num_encoder_layers=1,
        num_decoder_layers=1,
        num_heads=4,
        head_dim=8,
        dropout=0.0,
        max_source_length=8,
        max_target_length=8,
    )
    gqa = WMT14Transformer(attention_type='gqa', num_kv_heads=2, **common)
    gt_mha = WMT14Transformer(
        attention_type='gt_mha_quadratic',
        num_base_heads=2,
        num_generators=4,
        value_transform='lie',
        stabilize_generators=False,
        **common,
    )

    assert isinstance(gqa.decoder[0].cross_attention, GroupedQueryAttention)
    assert isinstance(gt_mha.decoder[0].cross_attention, LieGeneratedMetricAttention)

    # Cross-attention must support different decoder-query and encoder-memory
    # lengths; self-attention-only implementations can accidentally hide this.
    source = torch.tensor([[4, 5, 6, 2], [7, 8, 2, 3]])
    decoder_input = torch.tensor([[1, 9, 10], [1, 11, 3]])
    assert gqa(source, decoder_input).shape == (2, 3, 256)
    assert gt_mha(source, decoder_input).shape == (2, 3, 256)


class _LengthDataset:
    def __init__(self, size=100):
        self.source_lengths = np.arange(size, dtype=np.int64) % 17 + 1
        self.target_lengths = np.arange(size, dtype=np.int64) % 13 + 1

    def __len__(self):
        return len(self.source_lengths)


def test_ddp_bucket_shards_are_disjoint_with_shared_seed():
    dataset = _LengthDataset()
    rank0 = TokenBucketBatcher(dataset, tokens_per_batch=64, max_source_length=32, max_target_length=32, seed=19, rank=0, world_size=2, bucket_size=32)
    rank1 = TokenBucketBatcher(dataset, tokens_per_batch=64, max_source_length=32, max_target_length=32, seed=19, rank=1, world_size=2, bucket_size=32)

    indices0 = {index for batch in rank0.batches for index in batch}
    indices1 = {index for batch in rank1.batches for index in batch}
    assert indices0.isdisjoint(indices1)
    assert indices0 | indices1 == set(range(len(dataset)))
