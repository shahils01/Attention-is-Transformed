import hashlib,json,random
from pathlib import Path
ROOT=Path(__file__).resolve().parent; ENG=ROOT.parent
DATA=Path('/scratch/shahils/lgma_data/tinystory')
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for c in iter(lambda:f.read(8*1024*1024),b''):h.update(c)
 return h.hexdigest()
def canonical_hash(s):return hashlib.sha256(s.strip().encode()).hexdigest()
dev=json.loads((ENG/'development_prompts.json').read_text()); token=json.loads((ENG/'tokenizer.json').read_text())
for filename,expected in token['data_sha256'].items():assert digest(DATA/filename)==expected
stories=(DATA/'TinyStoriesV2-GPT4-valid.txt').read_text().split('<|endoftext|>')
dev_hashes={canonical_hash(stories[r['story_id']]) for r in dev['prompts']}; candidates={};exclusions={'short_or_no_boundary':0,'development':0,'validation_duplicate':0}
for i,s in enumerate(stories):
 cut=next((j for j in range(120,min(161,len(s))) if s[j].isspace()),None)
 if len(s)<256 or cut is None:exclusions['short_or_no_boundary']+=1;continue
 h=canonical_hash(s)
 if h in dev_hashes:exclusions['development']+=1;continue
 if h in candidates:exclusions['validation_duplicate']+=1;continue
 candidates[h]=(len(s),i,cut,h)
train_matches=set();train_count=0
with (DATA/'TinyStoriesV2-GPT4-train.txt').open(encoding='utf-8') as f:
 pending=''
 for chunk in iter(lambda:f.read(8*1024*1024),''):
  parts=(pending+chunk).split('<|endoftext|>');pending=parts.pop()
  for s in parts:
   train_count+=1; h=canonical_hash(s)
   if h in candidates:train_matches.add(h)
 if pending.strip():
  train_count+=1;h=canonical_hash(pending)
  if h in candidates:train_matches.add(h)
exclusions['exact_training_overlap_after_strip']=len(train_matches)
eligible=sorted(r for h,r in candidates.items() if h not in train_matches);rng=random.Random(20260919);rows=[]
for b in range(4):
 pool=eligible[len(eligible)*b//4:len(eligible)*(b+1)//4];assert len(pool)>=50
 for length,i,cut,h in rng.sample(pool,50):
  s=stories[i];rows.append({'prompt_id':f'eval_{len(rows)+1:03d}','story_id':i,'story_sha256':hashlib.sha256(s.encode()).hexdigest(),'canonical_story_sha256':h,'length_quartile':b+1,'story_characters':length,'prefix':s[:cut],'prefix_characters':cut,'reference_continuation':s[cut:]})
assert len(rows)==200 and not ({r['story_id'] for r in rows}&{r['story_id'] for r in dev['prompts']})
assert not ({r['canonical_story_sha256'] for r in rows}&train_matches)
record={'dataset_hashes':token['data_sha256'],'selection_seed':20260919,'train_stories_scanned':train_count,'eligible_stories':len(eligible),'exclusions':exclusions,'overlap_note':'Exact full-story comparison after stripping outer whitespace; no near-duplicate or prefix-overlap claim. Validation was used for checkpoint selection.','prompts':rows}
target=ROOT/'evaluation_prompts.json';assert not target.exists(),'Refusing to replace frozen prompts';target.write_text(json.dumps(record,ensure_ascii=False,indent=2))
config={'run_id':'tinystories-main-20260919-v1','prompt_file_sha256':digest(target),'checkpoint_manifest_sha256':digest(ENG/'verified_checkpoints.json'),'samples_per_prompt':5,'max_new_tokens':3000,'temperature':1.0,'top_k':20,'precision':'bf16','context_length':512,'stop_sequence':'<|endoftext|>','seed_master':20260919,'gpu_requirement':'A100 40GB PCIe','array_concurrency':8,'checkpoint_count':8,'generation_count':8000,'judge_status':'Not selected or invoked; do not inspect final generation quality before freezing judge configuration','code_sha256':{p.name:digest(p) for p in ROOT.glob('*.py')}}
(ROOT/'run_manifest.json').write_text(json.dumps(config,indent=2));print(json.dumps({k:v for k,v in record.items() if k!='prompts'},indent=2));print('PROMPTS_FROZEN',flush=True)
