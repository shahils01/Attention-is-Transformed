from pathlib import Path
import hashlib,json,os,sys,time
ROOT=Path(__file__).resolve().parent;ENG=ROOT.parent
sys.path.insert(0,str(ENG))
import engineer as eng
import torch

def main():
 index=int(sys.argv[1]); config=json.loads((ROOT/'run_manifest.json').read_text())
 assert eng.digest(Path(__file__))==config['code_sha256'][Path(__file__).name]
 assert eng.digest(ROOT/'evaluation_prompts.json')==config['prompt_file_sha256']
 assert eng.digest(ENG/'verified_checkpoints.json')==config['checkpoint_manifest_sha256']
 item=json.loads((ENG/'verified_checkpoints.json').read_text())['checkpoints'][index]
 assert eng.digest(Path(item['local_path']))==item['sha256']
 assert json.loads((ENG/'outputs'/f'{index:02d}'/'complete.json').read_text())['status']=='engineering_passed'
 previous=json.loads((ENG/'outputs'/f'{index:02d}'/'metadata.json').read_text());assert eng.digest(ENG/'engineer.py')==previous['runner_sha256']
 for name,h in previous['source_sha256'].items():
  if not Path(name).name.startswith('._'):assert eng.digest(ENG/name)==h
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 assert torch.cuda.is_bf16_supported();assert '40GB' in torch.cuda.get_device_name()
 token=json.loads((ENG/'tokenizer.json').read_text());vocab=token['vocabulary'];stoi={c:i for i,c in enumerate(vocab)}
 ckpt=torch.load(item['local_path'],map_location='cpu',weights_only=False,mmap=True)
 model=eng.TinyTransformerLM(vocab_size=len(vocab),**ckpt['model_config']);model.load_state_dict(ckpt['model_state'],strict=True);del ckpt
 model=model.cuda().eval();out=ROOT/'outputs'/f'{index:02d}';out.mkdir(parents=True,exist_ok=True)
 metadata={**previous,'gpu':torch.cuda.get_device_name(),'generation_config':config,'main_runner_sha256':eng.digest(Path(__file__)),'slurm_job_id':os.getenv('SLURM_JOB_ID'),'slurm_array_task_id':os.getenv('SLURM_ARRAY_TASK_ID')}
 path=out/'generations.jsonl';assert not path.exists(),'Refusing to overwrite results'
 (out/'metadata.json').write_text(json.dumps(metadata,indent=2))
 prompts=json.loads((ROOT/'evaluation_prompts.json').read_text())['prompts']
 # Warm up using the development set only; not included in timing or quality outputs.
 warm=json.loads((ENG/'development_prompts.json').read_text())['prompts'][0]['prefix'];eng.generate(model,stoi,vocab,warm,0,32)
 with path.open('x') as f:
  for p in prompts:
   for sample in range(config['samples_per_prompt']):
    seed=int.from_bytes(hashlib.sha256(f"{config['run_id']}:{config['seed_master']}:{p['prompt_id']}:{sample}".encode()).digest()[:4],'big')
    torch.cuda.synchronize();start=time.perf_counter()
    row=eng.generate(model,stoi,vocab,p['prefix'],seed,cap=config['max_new_tokens'])
    torch.cuda.synchronize();elapsed=time.perf_counter()-start
    row.update({'run_id':config['run_id'],'model':item['model'],'checkpoint_sha256':item['sha256'],'prompt_id':p['prompt_id'],'story_id':p['story_id'],'prompt':p['prefix'],'sample_index':sample,'end_to_end_seconds':elapsed,'operational_char_tokens_per_second':row['generated_tokens_including_stop']/elapsed,'within_context_excerpt':row['completion'][:320],'excerpt_evaluator_cut':len(row['completion'])>320,'gpu':torch.cuda.get_device_name()})
    f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush()
   print(json.dumps({'model':item['model'],'prompts_done':int(p['prompt_id'].split('_')[1]),'samples_done':int(p['prompt_id'].split('_')[1])*5}),flush=True)
 (out/'complete.json').write_text(json.dumps({'status':'generation_complete','samples':1000,'judge_status':'not run'},indent=2))
if __name__=='__main__':main()
