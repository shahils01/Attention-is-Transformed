"""Fixed-checkpoint engineering validation, not scientific quality evaluation."""
from contextlib import nullcontext
from pathlib import Path
import hashlib,json,os,platform,sys,time
import torch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'repo/src'))
from lgma.transformer import TinyTransformerLM

STOP='<|endoftext|>'
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def autocast(precision):
 return torch.autocast('cuda',dtype=torch.bfloat16) if precision=='bf16' else nullcontext()
def seed_for(prompt_id,sample=0):
 return int.from_bytes(hashlib.sha256(f'engineering-v1:20260919:{prompt_id}:{sample}'.encode()).digest()[:4],'big')
def next_logits(model,ids,cache=None):
 # Every supplied ids tensor contains the full narrative so far.
 if cache is not None and cache[0][0].shape[-2]<model.context_length:
  logits,cache=model(ids[:,-1:],past_key_values=cache,use_cache=True)
 elif ids.shape[1]<model.context_length:
  logits,cache=model(ids,use_cache=True)
 else:
  logits=model(ids[:,-model.context_length:]); cache=None
 return logits[:,-1,:],cache

def summarize_raw(raw,token_count,cap):
 found=raw.endswith(STOP)
 return {'completion':raw[:-len(STOP)] if found else raw,'finish_reason':'stop_sequence' if found else 'max_new_tokens','generated_tokens_including_stop':token_count,'cap':cap,'partial_stop_suffix':next((STOP[:n] for n in range(len(STOP)-1,0,-1) if raw.endswith(STOP[:n])),None) if not found else None}

@torch.inference_mode()
def generate(model,stoi,vocab,prompt,seed,cap=2048):
 torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
 ids=torch.tensor([[stoi[c] for c in prompt]],device='cuda'); initial=ids.shape[1]
 stop_ids=[stoi[c] for c in STOP]; generated=[]; cache=None; first_shift=None
 with autocast('bf16'):
  for n in range(cap):
   if ids.shape[1]>model.context_length and first_shift is None:first_shift=n
   logits,cache=next_logits(model,ids,cache)
   assert torch.isfinite(logits).all()
   # T=1, top-k=20, matching frozen development settings.
   cutoff=torch.topk(logits,20).values[:,-1:]
   logits=logits.masked_fill(logits<cutoff,-torch.inf)
   token=torch.multinomial(torch.softmax(logits,dim=-1),1)
   ids=torch.cat((ids,token),dim=1); generated.append(int(token.item()))
   if generated[-len(stop_ids):]==stop_ids:break
 raw=''.join(vocab[i] for i in generated)
 return {**summarize_raw(raw,len(generated),cap),'raw_completion':raw,'generated_token_ids':generated,'prompt_characters':initial,'first_shift_before_generated_token_zero_based':first_shift,'seed':seed,'temperature':1.0,'top_k':20,'precision':'bf16','context_length':model.context_length}

@torch.inference_mode()
def parity(model,stoi,text):
 # Real development story extended only if necessary to exercise the boundary.
 while len(text)<514:text+=' '+text
 ids=torch.tensor([[stoi[c] for c in text[:514]]],device='cuda')
 records=[]; precision_refs={}
 for precision in ['fp32','bf16']:
  with autocast(precision):
   cache=None
   for length in [127,128,511,512,513,514]:
    if length in [127,511]:
     # Initialize cache immediately before a single-token append.
     logits,cache=model(ids[:,:length],use_cache=True)
     actual=logits[:,-1,:]
    else:actual,cache=next_logits(model,ids[:,:length],cache)
    expected=model(ids[:,max(0,length-512):length])[:,-1,:]
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
    abs_error=float((actual.float()-expected.float()).abs().max())
    pa=actual.float().softmax(-1);pb=expected.float().softmax(-1)
    prob_error=float((pa-pb).abs().max())
    # Fixed engineering tolerances, logged before observing any checkpoint.
    limit=0.002 if precision=='fp32' else 0.03
    passed=prob_error<=limit
    records.append({'precision':precision,'length':length,'max_logit_error':abs_error,'max_probability_error':prob_error,'probability_tolerance':limit,'passed':passed,'cache_present':cache is not None})
    precision_refs[(precision,length)]=expected.float()
  if precision=='bf16':
   for length in [127,128,511,512,513,514]:
    p=precision_refs[('fp32',length)].softmax(-1);q=precision_refs[('bf16',length)].softmax(-1)
    err=float((p-q).abs().max())
    records.append({'precision':'bf16_vs_fp32','length':length,'max_probability_error':err,'probability_tolerance':0.05,'passed':err<=0.05})
 return records

def main():
 index=int(sys.argv[1]);out=ROOT/'outputs'/f'{index:02d}';out.mkdir(parents=True,exist_ok=True)
 manifest=json.loads((ROOT/'verified_checkpoints.json').read_text());item=manifest['checkpoints'][index]
 tokenizer=json.loads((ROOT/'tokenizer.json').read_text());vocab=tokenizer['vocabulary'];stoi={c:i for i,c in enumerate(vocab)}
 prompts=json.loads((ROOT/'development_prompts.json').read_text())['prompts']
 assert ''.join(vocab[stoi[c]] for c in STOP)==STOP
 assert summarize_raw('Story.'+STOP,18,18)['finish_reason']=='stop_sequence'
 assert summarize_raw('Story.<|en',10,10)['partial_stop_suffix']=='<|en'
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 torch.backends.cudnn.allow_tf32=False
 assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
 path=Path(item['local_path']);assert digest(path)==item['sha256']
 ckpt=torch.load(str(path),map_location='cpu',weights_only=False,mmap=True)
 config=dict(ckpt['model_config']);assert config['context_length']==512
 model=TinyTransformerLM(vocab_size=len(vocab),**config)
 model.load_state_dict(ckpt['model_state'],strict=True)
 metadata={**item,'checkpoint_step':ckpt.get('step'),'model_config':config,'parameters':sum(p.numel() for p in model.parameters()),'vocab_size':len(vocab),'torch_version':torch.__version__,'gpu':torch.cuda.get_device_name(),'python':platform.python_version(),'slurm_job_id':os.getenv('SLURM_JOB_ID'),'slurm_array_task_id':os.getenv('SLURM_ARRAY_TASK_ID'),'source_sha256':{str(p.relative_to(ROOT)):digest(p) for p in sorted((ROOT/'repo/src/lgma').glob('*.py'))},'runner_sha256':digest(Path(__file__)),'tokenizer_sha256':digest(ROOT/'tokenizer.json')}
 saved_args=ckpt.get('args',{})
 if not isinstance(saved_args,dict):saved_args=vars(saved_args) if hasattr(saved_args,'__dict__') else {}
 metadata['saved_training_arguments']={k:str(saved_args[k]) if isinstance(saved_args[k],Path) else saved_args[k] for k in ['seed','batch_size','grad_accum_steps','world_size','steps','data_path','val_data_path','precision'] if k in saved_args}
 del ckpt
 (out/'metadata.json').write_text(json.dumps(metadata,indent=2))
 model=model.cuda().eval()
 checks=parity(model,stoi,prompts[0]['prefix']+prompts[0]['reference_continuation'])
 (out/'parity.json').write_text(json.dumps(checks,indent=2))
 assert all(r['passed'] for r in checks),'Logit validation failed; inspect parity.json'
 p=prompts[0];a=generate(model,stoi,vocab,p['prefix'],seed_for(p['prompt_id']),64);b=generate(model,stoi,vocab,p['prefix'],seed_for(p['prompt_id']),64)
 assert a==b,'Repeated generation differed'
 output=out/'generations.jsonl'
 assert not output.exists(),'Refusing to overwrite existing generation records'
 with output.open('x') as f:
  for p in prompts:
   start=time.monotonic();row=generate(model,stoi,vocab,p['prefix'],seed_for(p['prompt_id']))
   row.update({'model':item['model'],'checkpoint_sha256':item['sha256'],'prompt_id':p['prompt_id'],'story_id':p['story_id'],'prompt':p['prefix'],'sample_index':0,'elapsed_seconds':time.monotonic()-start})
   f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush()
   print(json.dumps({k:row[k] for k in ['model','prompt_id','finish_reason','generated_tokens_including_stop','elapsed_seconds']}),flush=True)
 (out/'complete.json').write_text(json.dumps({'status':'engineering_passed','samples':len(prompts),'parity_passed':True,'reproducibility_passed':True},indent=2))
if __name__=='__main__':main()
