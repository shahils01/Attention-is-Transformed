"""Validate the historical BERT recipe, then launch its two-base ablation."""
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

STAGE = Path(__file__).resolve().parent
REPO = STAGE / "repo"
REFERENCE = Path('/work/hdd/biad/sshaik4/lgma_runs/bert_base_gt_mha_residual_b4g8h12_random_seed42_explicit_fuseqkv_v2')
RUN_NAME = 'bert_base_gt_mha_identity_qkv_b2g8h12_random_seed42_explicit_20260916'
OUTPUT = Path('/work/hdd/biad/sshaik4/lgma_runs') / RUN_NAME
reference = json.loads((STAGE / 'reference_manifest.json').read_text())['arguments']
requested = dict(reference)
requested.update(seed=42, num_base_heads=2, attention_type="gt_mha_identity_both", enforce_paper_gt_mha=False,
                 output_dir=str(OUTPUT), run_name=RUN_NAME)
argv = []
for key, value in requested.items():
    if value is None:
        continue
    flag = '--' + key.replace('_', '-')
    if isinstance(value, bool):
        if key == 'trust_remote_code' and not value:
            continue  # Historical parser uses store_true, with default False.
        argv.append(flag if value else '--no-' + key.replace('_', '-'))
    else:
        argv.extend([flag, str(value)])

sys.path.insert(0, str(REPO / 'src'))
entry = runpy.run_path(str(REPO / 'experiments/train_bert_mlm.py'))
original_argv = sys.argv[:]
sys.argv = ['train_bert_mlm.py', *argv]
args = entry['parse_args']()
sys.argv = original_argv
for key, value in requested.items():
    parsed = getattr(args, key)
    assert (str(parsed) if isinstance(parsed, Path) else parsed) == value, (key, parsed, value)
assert args.max_steps == 100000 and args.warmup_steps == 10000
assert args.per_device_train_batch_size * args.gradient_accumulation_steps * 4 == 256

import torch
import transformers
import datasets
from transformers import AutoTokenizer, TrainingArguments, Trainer, set_seed
from lgma.bert import load_bert_masked_lm, bert_parameter_counts, bert_optimizer_parameter_groups
from transformers.trainer_pt_utils import get_parameter_names

torch.set_num_threads(2)
set_seed(args.seed)
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, local_files_only=True)
model, audit = load_bert_masked_lm(
    args.model_name_or_path, attention_type=args.attention_type,
    initialization=args.initialization, num_base_heads=2, num_generators=8,
    generator_mixing=args.generator_mixing, use_sdpa=False,
    fuse_base_qkv=True, sdpa_gqa_mode='auto', enforce_paper_gt_mha=False,
)
reference_config = json.loads((STAGE / 'reference_config.json').read_text())
for key in ('hidden_size', 'num_hidden_layers', 'num_attention_heads', 'intermediate_size',
            'hidden_dropout_prob', 'attention_probs_dropout_prob', 'layer_norm_eps',
            'initializer_range', 'vocab_size', 'hidden_act', 'max_position_embeddings'):
    assert getattr(model.config, key) == reference_config[key], key
assert len(audit) == 12
for layer in model.bert.encoder.layer:
    attn = layer.attention.self.gt_attention
    assert (attn.num_base_heads, attn.num_heads, attn.num_generators, attn.base_dim, attn.value_dim) == (2, 12, 8, 64, 64)
    assert attn.theta_init == 'random_sphere' and attn.theta_init_scale == 0.02
    assert attn.generator_init_scale == 0.02 and attn.metric_mode == 'identity'
    assert attn.logit_scale_mode == 'sqrt_dim' and not attn.learn_head_temperature
    assert not attn.use_sdpa and attn.fuse_base_qkv
    assert attn.value_transform_mode == 'none'
groups, coordinates = bert_optimizer_parameter_groups(model, weight_decay=0.01, get_parameter_names=get_parameter_names)
assert len(coordinates) == 0
assert sorted({g['weight_decay'] for g in groups}) == [0.0, 0.01]
training_args = TrainingArguments(output_dir='/tmp/bert-b2-preflight-unused', use_cpu=True, report_to=[],
    learning_rate=args.learning_rate, weight_decay=args.weight_decay, max_steps=args.max_steps,
    warmup_steps=args.warmup_steps)
optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(training_args)
assert optimizer_kwargs['betas'] == (0.9, 0.999) and optimizer_kwargs['eps'] == 1e-8
assert optimizer_kwargs.get('fused') is True
assert str(training_args.lr_scheduler_type.value) == 'linear' and training_args.max_grad_norm == 1.0
model.train()
tokens = torch.randint(100, model.config.vocab_size, (2, 8))
loss = model(input_ids=tokens, attention_mask=torch.ones_like(tokens), labels=tokens).loss
assert torch.isfinite(loss)
loss.backward()
for layer in model.bert.encoder.layer:
    attn = layer.attention.self.gt_attention
    assert not hasattr(attn, "generators") and not hasattr(attn, "theta")
    assert torch.equal(attn.compute_metrics(), torch.eye(64).expand(12,64,64))
    assert torch.equal(attn.compute_value_transforms(), torch.eye(64).expand(12,64,64))
    assert not hasattr(attn, "value_generators") and not hasattr(attn, "value_theta")
    assert torch.equal(attn.compute_value_transforms(), torch.eye(64).expand(12,64,64))
# An optimizer step must leave every score transformation exactly identity.
optimizer = torch.optim.AdamW(groups, lr=args.learning_rate)
optimizer.step()
for layer in model.bert.encoder.layer:
    attn = layer.attention.self.gt_attention
    assert torch.equal(attn.compute_metrics(), torch.eye(64).expand(12,64,64))
    assert torch.equal(attn.compute_value_transforms(), torch.eye(64).expand(12,64,64))
del optimizer
raw = datasets.load_dataset(args.dataset_name, args.dataset_config)
assert len(raw['train']) > 0 and len(raw['validation']) > 0
baseline = json.loads((STAGE / 'baseline_preflight.json').read_text())
assert {k: raw[k]._fingerprint for k in ('train','validation')} == baseline['dataset_fingerprints']
differences = {k for k in requested if requested[k] != baseline['arguments'][k]}
assert differences <= {'attention_type', 'output_dir', 'run_name', 'seed'}, differences
assert {'attention_type', 'output_dir', 'run_name'} <= differences
assert bert_parameter_counts(model)['model_parameter_count'] == 91796538
report = dict(status='passed', torch=torch.__version__, transformers=transformers.__version__,
    datasets=datasets.__version__, parameters=bert_parameter_counts(model),
    dataset_fingerprints={k: raw[k]._fingerprint for k in ('train', 'validation')},
    optimizer=optimizer_cls.__name__, optimizer_kwargs=optimizer_kwargs,
    arguments=requested, forward_backward_loss=float(loss.detach()))
(STAGE / 'preflight.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2), flush=True)
del model, raw, loss, groups
if '--check-only' in original_argv:
    raise SystemExit(0)
if (OUTPUT / 'train_results.json').exists():
    raise SystemExit('Run already completed; refusing duplicate training')
if OUTPUT.exists():
    checkpoints = sorted(OUTPUT.glob('checkpoint-*'), key=lambda p: int(p.name.split('-')[-1]))
    assert checkpoints, 'Existing output has no resumable checkpoint; refusing overwrite'
    argv += ['--resume-from-checkpoint', str(checkpoints[-1])]
command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=4',
           'experiments/train_bert_mlm.py', *argv]
os.execv(sys.executable, command)
