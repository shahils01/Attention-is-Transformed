"""Read-only audit of supplied generation artifacts. No model or judge calls."""
import collections, hashlib, json, pathlib, random, statistics
SRC = pathlib.Path('/Users/shahilshaik/Downloads/tinystories_100_x5_1600')
OUT = pathlib.Path(__file__).resolve().parent

def read(name):
    return [json.loads(l) for l in (SRC / name).read_text().splitlines()]
def key(r): return r['model'], r['prompt_id'], r['sample_index']
def percentile(v, p):
    v=sorted(v); t=(len(v)-1)*p; i=int(t)
    return v[i]+(v[min(i+1,len(v)-1)]-v[i])*(t-i)
rows=read('completions.jsonl'); scores=read('blind_eval/unblinded_scores.jsonl')
raw={key(r):r for r in rows}; scored={key(r):r for r in scores}
assert len(raw)==len(rows)==len(scored)==len(scores)==4000
assert raw.keys()==scored.keys()
metrics=['grammar','consistency','creativity','plot']
for k,r in raw.items():
    assert r['completion']==scored[k]['completion']
    assert r['prompt']==scored[k]['prompt']
    assert r['generated_characters']==len(r['completion'])
    assert r['seed']==r['prompt_index']+100*r['sample_index']
    assert all(1<=scored[k][m]<=10 for m in metrics)
blinds={r['blind_id']:r for r in read('blind_eval/blind.jsonl')}
judgments={r['blind_id']:r for r in read('blind_eval/blind_scores.jsonl')}
for mapping in read('blind_eval/blind_mapping.jsonl'):
    b=blinds[mapping['blind_id']]; j=judgments[mapping['blind_id']]
    assert j['blind_record_sha256']==mapping['blind_record_sha256']
    bc={c['candidate_id']:c for c in b['candidates']}
    for c in mapping['candidates']:
        k=(c['model'],mapping['prompt_id'],mapping['sample_index'])
        assert bc[c['candidate_id']]['completion']==raw[k]['completion']
        assert hashlib.sha256(raw[k]['completion'].encode()).hexdigest()==c['completion_sha256']
        assert all(j['scores'][c['candidate_id']][m]==scored[k][m] for m in metrics)
models=sorted({r['model'] for r in rows}); prompts=sorted({r['prompt_id'] for r in rows})
u={r['prompt_id']:r for r in rows}
result={'source':str(SRC),'verified_records':len(rows),'models':{},'paired_bootstrap':{},'prompt_lengths':{'min':min(len(r['prompt']) for r in u.values()),'max':max(len(r['prompt']) for r in u.values()),'mean':statistics.mean(len(r['prompt']) for r in u.values())}}
for model in models:
    a=[r for r in rows if r['model']==model]; s=[r for r in scores if r['model']==model]
    result['models'][model]={'n':len(a),'mean_characters':statistics.mean(len(r['completion']) for r in a),'cap_hits':sum(len(r['completion'])==1600 for r in a),'over_512_combined':sum(len(r['prompt'])+len(r['completion'])>512 for r in a),'mean_score':statistics.mean(statistics.mean(r[m] for m in metrics) for r in s),'step':a[0]['checkpoint_step'],'parameters':a[0]['parameters'],'literal_stop_markers':sum('<|endoftext|>' in r['completion'] for r in a)}
def bootstrap(a,b,allowed=None,metric=None):
    ps=prompts if allowed is None else allowed
    ds=[]
    for p in ps:
        means=[]
        for model in [a,b]:
            rs=[scored[(model,p,i)] for i in range(5)]
            means.append(statistics.mean(r[metric] if metric else statistics.mean(r[m] for m in metrics) for r in rs))
        ds.append(means[0]-means[1])
    rng=random.Random(17029)
    bs=[statistics.mean(rng.choices(ds,k=len(ds))) for _ in range(20000)]
    return {'difference':statistics.mean(ds),'ci95':[percentile(bs,.025),percentile(bs,.975)],'prompts':len(ps)}
for b in ['mha','gqa']:
    result['paired_bootstrap']['residual_minus_'+b]=bootstrap('gt_mha_residual',b)
    for m in metrics:result['paired_bootstrap']['residual_minus_'+b+'_'+m]=bootstrap('gt_mha_residual',b,metric=m)
    clean=[p for p in prompts if all(len(raw[(model,p,i)]['completion'])<1600 for model in ['gt_mha_residual',b] for i in range(5))]
    result['paired_bootstrap']['residual_minus_'+b+'_exclude_any_capped_prompt']=bootstrap('gt_mha_residual',b,allowed=clean)
    clean=[p for p in prompts if u[p]['theme']!='age_appropriate_language']
    result['paired_bootstrap']['residual_minus_'+b+'_exclude_age_theme']=bootstrap('gt_mha_residual',b,allowed=clean)
result['capped_outputs']=[{'model':r['model'],'prompt_id':r['prompt_id'],'sample_index':r['sample_index'],'ending':r['completion'][-100:]} for r in rows if len(r['completion'])==1600]
result['bootstrap_method']='20,000 paired prompt-cluster percentile resamples; average five samples and four metrics within each prompt/model; Python random seed 17029. Conditional on saved generations and judge scores. Sensitivity subsets are exploratory.'
(OUT/'audit_results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='capped_outputs'},indent=2))
