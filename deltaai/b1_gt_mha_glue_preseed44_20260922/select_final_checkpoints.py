"""Select downstream checkpoints using dev scores only; never submit test results."""
import json
import math
from pathlib import Path

ROOT = Path('/work/hdd/biad/sshaik4/lgma_runs')
VERSION = 'glue_b1_gt_mha_preseed44_papersweep_20260922'
METHOD = 'gt_mha_residual'
results = {}
for task in ('cola', 'mnli', 'mrpc', 'qnli', 'qqp', 'rte', 'sst2', 'stsb', 'wnli'):
    metric = 'eval_matthews_correlation' if task == 'cola' else 'eval_combined_score' if task in ('mrpc', 'qqp', 'stsb') else 'eval_accuracy'
    prefix = f'{VERSION}_{task}'
    lr = '2e-5' if task == 'wnli' else json.loads((ROOT / f'{prefix}_selected_learning_rates.json').read_text())[METHOD]['selected_learning_rate']
    candidates = []
    for seed in ((42,) if task == 'wnli' else (42, 43, 44, 45, 46)):
        stage = 'fixed' if task == 'wnli' else 'grid' if seed <= 44 else 'final'
        path = ROOT / f'{prefix}_{METHOD}_lr{lr.replace("e-", "em")}_seed{seed}_{stage}'
        data = json.loads((path/'eval_results.json').read_text())
        assert (path/'model.safetensors').is_file()
        assert math.isfinite(data[metric]) and math.isfinite(data['eval_loss'])
        candidates.append(dict(seed=seed, path=str(path), score=data[metric], loss=data['eval_loss']))
    selected = max(candidates, key=lambda x: (x['score'], -x['loss'], -x['seed']))
    results[task] = dict(metric=metric, learning_rate=lr, selected=selected, candidates=candidates)
(ROOT / f'{VERSION}_selected_checkpoints.json').write_text(json.dumps(results, indent=2)+'\n')
print(json.dumps(results, indent=2))
