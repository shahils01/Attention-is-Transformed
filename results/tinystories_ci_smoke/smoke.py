import json, math, time, hashlib, gc, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from lgma.transformer import TinyTransformerLM

root=Path(__file__).resolve().parent
data=Path('/work/hdd/bifg/arai3/lgma_data/tinystories')
torch.set_num_threads(4)
torch.manual_seed(17029)
chars=set()
with (data/'TinyStoriesV2-GPT4-train.txt').open() as f:
    while chunk:=f.read(8*1024*1024): chars.update(chunk)
val=(data/'TinyStoriesV2-GPT4-valid.txt').read_text()
chars.update(val); vocab=sorted(chars); stoi={c:i for i,c in enumerate(vocab)}
stories=val.split('<|endoftext|>')
valid=[i for i,s in enumerate(stories) if len(s)>1]
ids=np.random.default_rng(17029).choice(valid,64,replace=False).tolist()
manifest={'seed':17029,'story_ids':ids,'validation_sha256':hashlib.sha256(val.encode()).hexdigest(),'vocabulary':vocab,'protocol':'64 random stories; story-isolated evaluation; contiguous nonoverlapping context windows; each within-story next-character target once; separators excluded; context resets every context_length characters; no padding; bf16; smoke only','total_stories':len(valid)}
(root/'manifest.json').write_text(json.dumps(manifest,indent=2))
outputs={}
for name in ['MHA','GT-MHA residual']:
    start=time.perf_counter()
    ckpt=torch.load(root/(name+'.pt'),map_location='cpu',weights_only=False,mmap=True)
    cfg=ckpt['model_config']
    model=TinyTransformerLM(vocab_size=len(vocab),**cfg)
    model.load_state_dict(ckpt['model_state'],strict=True)
    model=model.to('cuda').eval()
    params=sum(p.numel() for p in model.parameters())
    step=ckpt.get('step'); del ckpt; gc.collect()
    def loss_for(text):
        encoded=torch.tensor([stoi[c] for c in text],dtype=torch.long,device='cuda')
        total=0.; count=0
        for pos in range(0,len(text)-1,int(cfg['context_length'])):
            n=min(int(cfg['context_length']),len(text)-1-pos)
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                logits=model(encoded[pos:pos+n].unsqueeze(0))
                loss=F.cross_entropy(logits.float().reshape(-1,len(vocab)),encoded[pos+1:pos+n+1],reduction='sum')
            total+=loss.item(); count+=n
        return total,count
    a=loss_for(stories[ids[0]]); b=loss_for(stories[ids[0]])
    assert np.isclose(a[0],b[0],rtol=1e-6), (a,b)
    torch.cuda.synchronize(); eval_start=time.perf_counter()
    rows=[]
    for i in ids:
        loss,n=loss_for(stories[i]); assert math.isfinite(loss) and n==len(stories[i])-1
        rows.append({'story_id':i,'nll_sum':loss,'tokens':n})
    torch.cuda.synchronize(); elapsed=time.perf_counter()-eval_start
    (root/(name+'_per_story.json')).write_text(json.dumps(rows,indent=2))
    losses=np.array([r['nll_sum'] for r in rows]); counts=np.array([r['tokens'] for r in rows])
    indices=np.random.default_rng(731).integers(0,len(rows),(2000,len(rows)))
    boot=losses[indices].sum(1)/counts[indices].sum(1)
    mean=losses.sum()/counts.sum(); ci=np.quantile(boot,[.025,.975])
    outputs[name]={'checkpoint_step':step,'config':cfg,'parameters':params,'nll':mean,'perplexity':math.exp(mean),'nll_ci95':ci.tolist(),'ppl_ci95':np.exp(ci).tolist(),'evaluated_tokens':int(counts.sum()),'eval_seconds':elapsed,'tokens_per_second':int(counts.sum())/elapsed,'load_and_eval_seconds':time.perf_counter()-start,'repeatability_passed':True}
    outputs[name]['bootstrap_nll']=boot.tolist()
    print(json.dumps({name:{k:v for k,v in outputs[name].items() if k not in ['bootstrap_nll','config']}}),flush=True)
    del model; gc.collect(); torch.cuda.empty_cache()
a=np.array(outputs['MHA'].pop('bootstrap_nll')); b=np.array(outputs['GT-MHA residual'].pop('bootstrap_nll'))
outputs['paired_difference_GT_minus_MHA']={'nll_ci95':np.quantile(b-a,[.025,.975]).tolist(),'ppl_ci95':np.quantile(np.exp(b)-np.exp(a),[.025,.975]).tolist()}
outputs['status']='SMOKE_PASSED_NOT_PAPER_RESULTS'
(root/'summary.json').write_text(json.dumps(outputs,indent=2))
print(outputs['status'],flush=True)
