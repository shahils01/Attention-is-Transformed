import os, json
from pathlib import Path
import torch
from transformers import set_seed
from transformers.trainer_pt_utils import get_parameter_names
from lgma.bert import load_bert_sequence_classifier, bert_optimizer_parameter_groups
stage=Path(__file__).resolve().parent
assert torch.cuda.device_count()==2
assert torch.cuda.is_bf16_supported()
results=[]
for gpu in range(2):
 torch.cuda.set_device(gpu)
 prop=torch.cuda.get_device_properties(gpu)
 assert 'A100' in prop.name and prop.total_memory>75*1024**3
 set_seed(42)
 model,_=load_bert_sequence_classifier(str(stage/'checkpoint'),num_labels=2,attention_type='gt_mha_identity_both',num_base_heads=4,num_generators=8,fuse_base_qkv=True,enforce_paper_gt_mha=False)
 model.to(f'cuda:{gpu}').train()
 groups,_=bert_optimizer_parameter_groups(model,weight_decay=.01,get_parameter_names=get_parameter_names)
 optimizer=torch.optim.AdamW(groups,lr=2e-5,betas=(.9,.999),eps=1e-8,fused=True)
 tokens=torch.randint(100,model.config.vocab_size,(2,16),device=f'cuda:{gpu}')
 with torch.autocast('cuda',dtype=torch.bfloat16):
  loss=model(input_ids=tokens,attention_mask=torch.ones_like(tokens),labels=torch.tensor([0,1],device=f'cuda:{gpu}')).loss
 assert torch.isfinite(loss)
 loss.backward()
 torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
 optimizer.step()
 assert all(torch.isfinite(p).all() for p in model.parameters())
 results.append({'gpu':gpu,'name':prop.name,'loss':loss.item(),'bf16_native_gqa_fused_adamw':'passed'})
 del model,optimizer,groups,loss
 torch.cuda.empty_cache()
(stage/'gpu_smoke.json').write_text(json.dumps(results,indent=2))
print(results,flush=True)
