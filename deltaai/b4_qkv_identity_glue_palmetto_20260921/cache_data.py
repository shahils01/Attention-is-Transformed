import datasets,evaluate,json
from pathlib import Path
info={}
for task in ('cola','mrpc','rte','mnli','qnli','sst2','qqp','stsb','wnli'):
 d=datasets.load_dataset('glue',task)
 m=evaluate.load('glue',task)
 info[task]={k:{'rows':len(v),'fingerprint':v._fingerprint} for k,v in d.items()}
 print(task,info[task],flush=True)
Path(__file__).with_name('dataset_manifest.json').write_text(json.dumps(info,indent=2))
