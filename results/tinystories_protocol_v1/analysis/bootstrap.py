"""Paired prompt-cluster percentile bootstrap of fixed-checkpoint judge scores."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'judged_scores.json'
data = json.loads(SOURCE.read_text())
rows = data['rows']
metrics = ['grammar', 'consistency', 'creativity', 'plot', 'average']
models = sorted({r['model'] for r in rows})
prompts = sorted({r['prompt_id'] for r in rows})
assert len(rows) == 8000 and len(models) == 8 and len(prompts) == 200
index = {}
for r in rows:
    key = (r['model'], r['prompt_id'], r['sample_index'])
    assert key not in index
    assert all(type(r['scores'][k]) is int and 1 <= r['scores'][k] <= 10 for k in metrics[:4])
    index[key] = r
# Average five generations within each prompt, preserving the prompt as cluster.
x = np.empty((200, 8, 5), dtype=np.float64)
for mi, model in enumerate(models):
    assert len({r['checkpoint_sha256'] for r in rows if r['model'] == model}) == 1
    for pi, prompt in enumerate(prompts):
        rs = [index[(model, prompt, s)] for s in range(5)]
        values = np.array([[r['scores'][k] for k in metrics[:4]] for r in rs])
        x[pi, mi, :4] = values.mean(axis=0)
        x[pi, mi, 4] = values.mean()
point = x.mean(axis=0)
# Verify the prompt-weighted estimator equals the raw mean in this balanced design.
for mi, model in enumerate(models):
    raw = np.array([[r['scores'][k] for k in metrics[:4]] for r in rows if r['model'] == model])
    assert np.allclose(point[mi, :4], raw.mean(axis=0), atol=1e-12)
seed, n_boot = 20260920, 20000
rng = np.random.default_rng(seed)
boot = np.empty((n_boot, 8, 5))
for start in range(0, n_boot, 200):
    ids = rng.integers(0, len(prompts), size=(min(200, n_boot-start), len(prompts)))
    boot[start:start+len(ids)] = x[ids].mean(axis=1)
def stats(est, reps):
    lo, hi = np.quantile(reps, [0.025, 0.975], axis=0, method='linear')
    return {k: {'estimate':float(est[i]), 'ci95':[float(lo[i]), float(hi[i])]} for i,k in enumerate(metrics)}
results = {m:stats(point[i], boot[:,i]) for i,m in enumerate(models)}
pairs = {}
for a,b in [(m,'MHA') for m in models if m!='MHA'] + [('GT-MHA residual',m) for m in models if m not in ('MHA','GT-MHA residual')]:
    i,j=models.index(a),models.index(b)
    pairs[a+' minus '+b] = stats(point[i]-point[j], boot[:,i]-boot[:,j])
report = {'method':'Paired prompt-cluster percentile bootstrap; five generations retained per sampled prompt; identical prompt draws across all methods',
    'bootstrap_replicates':n_boot,'random_seed':seed,'prompts':200,'samples_per_prompt':5,'models':8,
    'interval_scope':'Pointwise 95% intervals, no multiplicity correction. Prompt-sampling uncertainty conditional on fixed checkpoints, sampled completions and recorded judge outputs. Not training-seed variability, judge repeatability, or a formal equivalence test.',
    'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'numpy_version':np.__version__,
    'model_scores':results,'paired_differences':pairs}
(ROOT/'bootstrap_results.json').write_text(json.dumps(report,indent=2)+'\n')
lines=['# TinyStories generation: paired prompt bootstrap','',report['method']+'.',
       f'20,000 resamples; seed {seed}; 200 prompts × five generations per model.','',report['interval_scope'],'',
       '| Model | Grammar | Consistency | Creativity | Plot | Average [95% CI] |',
       '|---|---:|---:|---:|---:|---:|']
for m in sorted(models,key=lambda m:results[m]['average']['estimate'],reverse=True):
    s=results[m];a=s['average'];lo,hi=a['ci95']
    lines.append('| '+m+' | '+' | '.join(f"{s[k]['estimate']:.3f}" for k in metrics[:4])+f" | {a['estimate']:.3f} [{lo:.3f}, {hi:.3f}] |")
lines += ['','## Paired average-score differences','','Positive differences favor the first model.','', '| Comparison | Difference | 95% CI |','|---|---:|---:|']
for p,s in pairs.items():
    a=s['average'];lo,hi=a['ci95'];lines.append(f"| {p} | {a['estimate']:+.4f} | [{lo:+.4f}, {hi:+.4f}] |")
lines += ['','Per-criterion intervals and paired differences are in `bootstrap_results.json`. All 8,000 unique records passed balance and score-range checks.']
(ROOT/'bootstrap_report.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
