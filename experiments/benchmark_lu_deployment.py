"""Compare compact LU, exported LU and native GQA4 on one GPU."""
import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lgma.lu_export import export_lu_model
from lgma.packed_gqa import pack_gqa_model
from lgma.transformer import TinyTransformerLM


def make_model(architecture):
    return TinyTransformerLM(vocab_size=230, d_model=1024, num_layers=12, num_heads=16,
        head_dim=64, num_base_heads=4, num_kv_heads=4, attention_type=architecture,
        num_generators=0, context_length=512, dropout=0.1).cuda()


@torch.inference_mode()
def fidelity(models, tokens, precision):
    def outputs(model):
        prompt, continuation = tokens[:, :496], tokens[:, 496:512]
        cache = model.allocate_kv_cache(tokens.size(0), 512)
        prefill, _ = model(prompt, past_key_values=cache, use_cache=True)
        steps = [model(token, past_key_values=cache, use_cache=True)[0]
                 for token in continuation.split(1, dim=1)]
        return {'prefill': prefill, 'decode': torch.cat(steps, dim=1)}

    scores = {name: outputs(model) for name, model in models.items()}
    records = {}
    for candidate, reference in [('export', 'compact'), ('export_packed', 'compact'), ('gqa4_packed', 'gqa4')]:
        records[candidate] = {}
        for phase in ('prefill', 'decode'):
            a = scores[reference][phase].float()
            b = scores[candidate][phase].float()
            start = 1 if phase == 'prefill' else 497
            targets = tokens[:, start:start+a.size(1)].reshape(-1)
            logp, logq = a.log_softmax(-1), b.log_softmax(-1)
            rel = ((b-a).norm() / a.norm()).item()
            kl = (logp.exp() * (logp-logq)).sum(-1).mean().item()
            nll_delta = (F.cross_entropy(b.flatten(0, 1), targets)
                         - F.cross_entropy(a.flatten(0, 1), targets)).abs().item()
            if not torch.isfinite(b).all() or not torch.isfinite(a).all():
                raise RuntimeError(f"Nonfinite logits in {candidate}/{phase}")
            if precision == 'fp32':
                torch.testing.assert_close(b, a, atol=1e-4, rtol=1e-4)
                if nll_delta > 1e-5:
                    raise RuntimeError(f"FP32 NLL mismatch {candidate}/{phase}: {nll_delta}")
            elif rel > 1/64 or kl > 0.001 or nll_delta > 0.002:
                raise RuntimeError(f"BF16 fidelity failed {candidate}/{phase}: {rel=} {kl=} {nll_delta=}")
            records[candidate][phase] = dict(relative_l2=rel, mean_kl=kl, abs_mean_nll_delta=nll_delta,
                                               max_abs=(a-b).abs().max().item())
    return records


@torch.inference_mode()
def timings(models, batch, tokens):
    names = list(models)
    results = {}
    caches = {n: model.allocate_kv_cache(batch, 512) for n, model in models.items()}
    for phase, warmup, repeats, length in [('prefill', 10, 60, 512), ('decode', 5, 30, 16)]:
        samples = {n: [] for n in names}
        for i in range(warmup + repeats):
            order = names[i % len(names):] + names[:i % len(names)]
            if (i // len(names)) % 2:
                order.reverse()
            for name in order:
                model = models[name]
                if phase == 'decode':
                    cache = caches[name]
                    for layer in cache:
                        layer.reset()
                    model(tokens[:, :496], past_key_values=cache, use_cache=True)
                torch.cuda.synchronize()
                start = time.perf_counter()
                if phase == 'prefill':
                    cache = model.allocate_kv_cache(batch, 512)
                    model(tokens, past_key_values=cache, use_cache=True)
                else:
                    for index in range(496, 512):
                        model(tokens[:, index:index+1], past_key_values=cache, use_cache=True)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                if i >= warmup:
                    samples[name].append(elapsed)
        results[phase] = {
            name: {
                'seconds': times,
                'tokens_per_sec': batch * length / statistics.median(times),
            }
            for name, times in samples.items()
        }
    results['kv_bytes'] = {n: sum(x.allocated_bytes for x in c) for n, c in caches.items()}
    assert len(set(results['kv_bytes'].values())) == 1
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(73)
    torch.backends.cuda.matmul.allow_tf32 = False
    compact = make_model('gt_mha_lu')
    # Exercise nonzero updates. Synthetic data provides reproducible weights and inputs.
    opt = torch.optim.AdamW(compact.parameters(), lr=3e-4)
    for _ in range(3):
        data = torch.randint(230, (4, 513), device='cuda')
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            _, loss = compact(data[:, :-1], targets=data[:, 1:])
        loss.backward()
        opt.step()
    opt.zero_grad(set_to_none=True)
    del opt
    compact.eval()
    exported = export_lu_model(compact)
    native = make_model('gqa').eval()
    models = dict(compact=compact, export=exported, gqa4=native,
                  export_packed=pack_gqa_model(exported), gqa4_packed=pack_gqa_model(native))
    val = torch.randint(230, (8, 513), device='cuda')
    record = {
        'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'gpu': torch.cuda.get_device_name(), 'torch': str(torch.__version__), 'cuda': torch.version.cuda,
        'timed_dtype': 'bfloat16', 'seed': 73, 'tf32': False,
        'prefill_tokens': 512, 'decode_prefix': 496, 'decode_tokens': 16,
        'synthetic_updates': 3,
        'validation_inputs_sha256': hashlib.sha256(val.cpu().numpy().tobytes()).hexdigest(),
        'parameters': {name: sum(p.numel() for p in model.parameters()) for name, model in models.items()},
        'fidelity': {}, 'results': {},
    }
    record['fidelity']['fp32'] = fidelity(models, val, 'fp32')
    for model in models.values():
        model.bfloat16()
    record['fidelity']['bf16'] = fidelity(models, val, 'bf16')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for batch in (4, 16):
        tokens = torch.randint(230, (batch, 512), device='cuda')
        result = timings(models, batch, tokens)
        record['results'][batch] = result
        args.output.write_text(json.dumps(record, indent=2) + '\n')
        print(batch, {p:{n:round(v['tokens_per_sec']) for n,v in result[p].items()}
                      for p in ('prefill','decode')}, flush=True)


if __name__ == '__main__':
    main()
