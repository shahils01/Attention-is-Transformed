import concurrent.futures, hashlib, json, os, urllib.parse, urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parent
manifest=json.loads((ROOT/'checkpoint_manifest.json').read_text())
(ROOT/'checkpoints').mkdir(exist_ok=True)
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
 return h.hexdigest()
def stage(item):
 target=ROOT/'checkpoints'/(item['sha256']+'.pt')
 if not target.exists():
  url='https://huggingface.co/'+manifest['repository']+'/resolve/'+manifest['revision']+'/'+urllib.parse.quote(item['path'],safe='/')
  tmp=target.with_suffix('.partial')
  urllib.request.urlretrieve(url,tmp)
  assert sha(tmp)==item['sha256'],item['model']
  os.replace(tmp,target)
 assert target.stat().st_size==item['size_bytes'] and sha(target)==item['sha256']
 print('VERIFIED '+item['model'],flush=True)
 return {**item,'local_path':str(target),'verified_sha256':item['sha256']}
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
 verified=list(pool.map(stage,manifest['checkpoints']))
(ROOT/'verified_checkpoints.json').write_text(json.dumps({**manifest,'checkpoints':verified},indent=2))
data=Path('/scratch/shahils/lgma_data/tinystory'); vocab=set(); hashes={}
for name in ['TinyStoriesV2-GPT4-train.txt','TinyStoriesV2-GPT4-valid.txt']:
 path=data/name; hashes[name]=sha(path)
 with path.open(encoding='utf-8') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),''):vocab.update(chunk)
expected=json.loads((ROOT/'development_prompts.json').read_text())['source_sha256']
assert hashes['TinyStoriesV2-GPT4-valid.txt']==expected
(ROOT/'tokenizer.json').write_text(json.dumps({'vocabulary':sorted(vocab),'data_sha256':hashes},ensure_ascii=False,indent=2))
print('STAGING_COMPLETE',flush=True)
