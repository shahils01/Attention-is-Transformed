"""Small CPU preflight; does not create W&B runs or training allocations."""
import io
import torch
from lgma.transformer import TinyTransformerLM
from lgma.identity import QKIdentityAttention
from lgma.synthetic import CharTokenizer, make_lm_batch
from ospool.train_tinystories_compat import compact_encode
from experiments.train_tinystories_qk_identity import compact_batch as compact_make_lm_batch

torch.set_num_threads(2)
kw = dict(vocab_size=23, d_model=32, num_layers=2, num_heads=8,
          head_dim=4, base_dim=4, value_dim=4, num_base_heads=4,
          num_generators=8, context_length=8, dropout=0.,
          value_transform="lie", theta_init="random_sphere",
          theta_init_scale=.02, generator_init_scale=.02,
          stabilize_generators=False)
torch.manual_seed(0)
ref = TinyTransformerLM(attention_type="lgma_residual", **kw)
torch.manual_seed(0)
model = TinyTransformerLM(attention_type="lgma_qk_identity", **kw)
for name, parameter in model.named_parameters():
    assert torch.equal(parameter, dict(ref.named_parameters())[name]), name
for m in model.modules():
    if isinstance(m, QKIdentityAttention):
        assert "theta" not in dict(m.named_parameters())
        assert "generators" not in dict(m.named_parameters())
        assert m.value_transform_mode == "residual" and m.learn_head_temperature
        assert torch.equal(m.compute_metrics(), torch.eye(4).expand(8,4,4))
inputs = torch.randint(0,23,(2,8))
result = model(inputs)
if isinstance(result, tuple):
    result = result[0]
result.square().mean().backward()
for name, p in model.named_parameters():
    assert p.grad is not None and torch.isfinite(p.grad).all(), name
for m in model.modules():
    if isinstance(m,QKIdentityAttention):
        assert m.value_generators.grad.abs().sum() > 0
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
optimizer.step()
for m in model.modules():
    if isinstance(m,QKIdentityAttention):
        assert torch.equal(m.compute_metrics(), torch.eye(4).expand(8,4,4))
buffer = io.BytesIO()
torch.save(model.state_dict(), buffer)
buffer.seek(0)
restored = TinyTransformerLM(attention_type="lgma_qk_identity", **kw)
restored.load_state_dict(torch.load(buffer, weights_only=True))
torch.testing.assert_close(model(inputs),restored(inputs))
tokenizer = CharTokenizer("abcde" * 20)
encoded = tokenizer.encode("abcde" * 20)
compact = compact_encode(tokenizer,"abcde" * 20)
torch.manual_seed(42)
a = make_lm_batch(encoded,3,8)
torch.manual_seed(42)
b = compact_make_lm_batch(compact,3,8)
assert torch.equal(a.input_ids,b.input_ids) and torch.equal(a.targets,b.targets)
g1 = torch.Generator().manual_seed(123)
g2 = torch.Generator().manual_seed(123)
a = make_lm_batch(encoded,3,8,generator=g1)
b = compact_make_lm_batch(compact,3,8,generator=g2)
assert torch.equal(a.input_ids,b.input_ids) and torch.equal(a.targets,b.targets)
assert torch.equal(g1.get_state(),g2.get_state())
print("PASS: retained initialization, identity invariance, learned V gradients, checkpoint roundtrip, compact sampler parity", flush=True)

import sys
from experiments.train_tinystories_qk_identity import install
training = install()
sys.argv = [__file__, "--data_path", __file__, "--device", "cpu",
            "--steps", "2", "--batch_size", "2", "--d_model", "32",
            "--num_layers", "1", "--num_heads", "8", "--head_dim", "4",
            "--num_base_heads", "4", "--context_length", "8",
            "--value_transform", "lie", "--attention", "lgma_qk_identity",
            "--diagnostic_every", "1", "--diagnostic_batches", "1",
            "--eval_every", "1", "--eval_batches", "1", "--log_every", "1",
            "--wandb_mode", "disabled"]
training.main()
print("PASS: real training loop and diagnostics", flush=True)
