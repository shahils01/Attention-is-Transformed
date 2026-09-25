import json, torch
from pathlib import Path
p=Path('/work/hdd/bicn/anayak2/lgma_runs/tinystories_qkv_identity_b4g8h16_seed0_20260910/checkpoint_step_96800.pt')
torch.set_num_threads(2)
c=torch.load(p,map_location='cpu',weights_only=False)
assert c['step']==96800
assert c['optimizer_state']['state']
assert len(c['data_generator_states'])==4
for state in c['data_generator_states']:
 g=torch.Generator();g.set_state(state.cpu())
a=c['args']
for k,v in {'attention':'lgma_qkv_identity','num_base_heads':4,'num_heads':16,'num_layers':12,'d_model':1024,'batch_size':256,'grad_accum_steps':2,'steps':250000,'seed':0,'context_length':512,'lr':3e-4,'lr_hold_steps':50000,'min_lr':1e-4}.items():
 assert a[k]==v,(k,a[k],v)
steps=sorted({int(v['step']) for v in c['optimizer_state']['state'].values()})
assert steps==[96800],steps
buffers=[(k,v) for k,v in c['model_state'].items() if k.split('.')[-1] in ('theta','generators','value_theta','value_generators')]
assert len(buffers)==48
assert all(torch.count_nonzero(v)==0 for _,v in buffers)
r={'step':c['step'],'optimizer_steps':steps,'sampler_states':4,'identity_buffers_verified':len(buffers),'remaining_steps':250000-c['step'],'source_job':15776757}
Path(__file__).with_name('checkpoint_verified.json').write_text(json.dumps(r,indent=2))
print(r,flush=True)
