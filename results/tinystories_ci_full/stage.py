import json,hashlib,subprocess,urllib.request,urllib.parse,os
from pathlib import Path
root=Path(__file__).resolve().parent
repo='shahils/GT-MHA'
info=json.load(urllib.request.urlopen('https://huggingface.co/api/models/'+repo)); revision=info['sha']
base='TinyStories-v2 Checkpoints/'
files=json.load(urllib.request.urlopen('https://huggingface.co/api/models/'+repo+'/tree/'+revision+'/'+urllib.parse.quote(base.rstrip('/'),safe='/')+'?recursive=true&limit=1000'))
manifest={'repository':repo,'revision':revision,'files':{}}
for method in ['MHA','GQA','MQA','Collaborative MHA','GT-MHA residual']:
    matches=[f for f in files if f['type']=='file' and f['path'].startswith(base+method+'/') and f['path'].endswith('.pt')]
    assert len(matches)==1,(method,matches)
    f=matches[0]; expected=f['lfs']['oid']; target=root/(method+'.pt')
    if method=='GT-MHA residual': assert expected=='a3b36ada91bb42b22b6df4da2b36be5cd3333f5a473e857c326aa06323ae06f2'
    if method=='MHA' and not target.exists():
        os.link('/work/hdd/bifg/arai3/lgma_jobs/tinystories_ci_smoke_20260915/MHA.pt',target)
    if not target.exists():
        url='https://huggingface.co/'+repo+'/resolve/'+revision+'/'+urllib.parse.quote(f['path'],safe='/')
        partial=str(target)+'.partial'
        subprocess.run(['curl','-L','--fail','--silent','--show-error','--retry','2','--max-time','600','-o',partial,url],check=True)
        os.rename(partial,target)
    h=hashlib.sha256()
    with target.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''): h.update(chunk)
    assert h.hexdigest()==expected,method
    manifest['files'][method]={'path':f['path'],'sha256':expected,'size':target.stat().st_size}
    (root/'checkpoint_manifest.json').write_text(json.dumps(manifest,indent=2))
    print('Verified '+method,flush=True)
print('ALL_CHECKPOINTS_VERIFIED',flush=True)
