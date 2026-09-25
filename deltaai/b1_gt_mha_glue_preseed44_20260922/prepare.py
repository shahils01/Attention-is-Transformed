import hashlib
import json
import math
import os
from pathlib import Path
import runpy
import sys
import tempfile
import urllib.parse
import urllib.request

STAGE = Path(__file__).resolve().parent
CHECKPOINT = STAGE / 'checkpoint'
expected = json.loads((STAGE / 'source_checkpoint_hashes.json').read_text())
for name, digest in expected.items():
    assert hashlib.sha256((CHECKPOINT/name).read_bytes()).hexdigest() == digest, name
manifest = json.loads((CHECKPOINT/'bert_gt_mha_manifest.json').read_text())
assert manifest['arguments']['seed'] == 44
assert manifest['attention_type'] == 'gt_mha_residual'
assert (manifest['num_base_heads'], manifest['num_generators']) == (1, 8)
print('PASS: selected seed-44 GT-MHA residual checkpoint hashes and manifest', flush=True)

sys.path.insert(0, str(STAGE / 'repo/src'))
import torch
import datasets
import evaluate
import transformers
from safetensors.torch import load_model
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer, set_seed
from lgma.bert import replace_bert_self_attention, load_bert_sequence_classifier

torch.set_num_threads(2)
set_seed(42)
config = AutoConfig.from_pretrained(CHECKPOINT, local_files_only=True)
assert config.num_attention_heads == 12 and config.num_hidden_layers == 12
model = AutoModelForMaskedLM.from_config(config)
replace_bert_self_attention(model, attention_type='gt_mha_residual', num_base_heads=1,
    num_generators=8, generator_mixing='softmax', use_sdpa=False,
    fuse_base_qkv=True, sdpa_gqa_mode='auto', initialize_from_mha=False,
    enforce_paper_gt_mha=False)
load_model(model, str(CHECKPOINT / 'model.safetensors'), strict=True)
assert sum(p.numel() for p in model.parameters()) == 90813498
# The historical loader requires this filename and tied-weight-complete state dict.
# The original uploaded safetensors file remains unchanged.
torch.save(model.state_dict(), CHECKPOINT / 'gt_mha_state_dict.pt')
classifier, audit = load_bert_sequence_classifier(str(CHECKPOINT), num_labels=2,
    attention_type='gt_mha_residual', num_base_heads=1, num_generators=8,
    fuse_base_qkv=True, enforce_paper_gt_mha=False)
assert len(audit) == 12
for component in ('embeddings', 'encoder'):
    expected = getattr(model.bert, component).state_dict()
    actual = getattr(classifier.bert, component).state_dict()
    assert expected.keys() == actual.keys()
    for name in expected:
        assert torch.equal(expected[name], actual[name]), name
tokens = torch.randint(100, config.vocab_size, (2, 8))
loss = classifier(input_ids=tokens, attention_mask=torch.ones_like(tokens), labels=torch.tensor([0, 1])).loss
assert torch.isfinite(loss)
loss.backward()
assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in classifier.bert.encoder.parameters())
del classifier, model, loss
print('PASS: exact pretrained backbone restoration and finite classifier backward', flush=True)

cache = {}
for task in ('cola', 'mnli', 'mrpc', 'qnli', 'qqp', 'rte', 'sst2', 'stsb', 'wnli'):
    raw = datasets.load_dataset('glue', task)
    metric = evaluate.load('glue', task)
    preds = [0.1, 0.9, 0.4] if task == 'stsb' else [0, 1, 0]
    values = metric.compute(predictions=preds, references=preds)
    assert all(math.isfinite(float(v)) for v in values.values())
    validation = 'validation_matched' if task == 'mnli' else 'validation'
    cache[task] = dict(train=len(raw['train']), validation=len(raw[validation]))
    assert cache[task]['train'] and cache[task]['validation']
    print('PASS: GLUE/' + task + ' dataset and metric cached', flush=True)

# Exercise the actual generic selector, including its smaller-LR tie break.
selector = STAGE / 'select_glue_learning_rates.py'
with tempfile.TemporaryDirectory(prefix='b2-glue-selector-') as folder:
    root = Path(folder)
    for key in ('eval_matthews_correlation', 'eval_combined_score', 'eval_accuracy'):
        for lr, score in [('2em5', .8), ('3em5', .8), ('5em5', .7)]:
            for seed in (42, 43, 44):
                out = root / f'test_gt_mha_residual_lr{lr}_seed{seed}_grid'
                out.mkdir(exist_ok=True)
                (out / 'eval_results.json').write_text(json.dumps({key: score}))
        sys.argv = [str(selector), '--root', str(root), '--prefix', 'test', '--metric-key', key,
            '--methods', 'gt_mha_residual', '--output-json', str(root/'result.json'), '--output-tsv', str(root/'result.tsv')]
        runpy.run_path(str(selector), run_name='__main__')
        assert json.loads((root/'result.json').read_text())['gt_mha_residual']['selected_learning_rate'] == '2e-5'
(STAGE/'preflight.json').write_text(json.dumps(dict(status='passed', pretrained_seed=44,
    parameters=90813498, exact_backbone_restoration=True, glue_cache=cache,
    torch=torch.__version__, transformers=transformers.__version__, datasets=datasets.__version__), indent=2))
print('PASS: preparation and selector checks complete', flush=True)
