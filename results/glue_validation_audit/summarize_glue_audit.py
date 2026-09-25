import json,pathlib,re,statistics
r=pathlib.Path('results/glue_validation_audit')
records=sum((json.loads((r/f'{a}.json').read_text()) for a in ['anayak2','arai3','sshaik4']),[])
manifest={}
for f in ['selected_mha_gt.json','selected_baselines.json']: manifest.update(json.loads((r/f).read_text()))
byname={}
for x in records: byname.setdefault(x['run_name'],[]).append(x)
summary=[]; issues=[]; chosen=[]
for method in ['mha','gqa','collaborative','gt_mha_residual']:
 for task,s in manifest[method].items():
  basename=pathlib.Path(s['checkpoint_dir']).name
  names=[basename] if task=='wnli' else [re.sub(r'_seed\d+_(grid|final)$',f'_seed{seed}_'+('grid' if seed<=44 else 'final'),basename) for seed in range(42,47)]
  vals=[]; runrows=[]
  metric=s['validation_metric']
  for name in names:
   hits=byname.get(name,[])
   if not hits: issues.append('Missing '+name);continue
   if len(hits)>1 and any(h['metrics']!=hits[0]['metrics'] for h in hits): issues.append('Conflicting duplicates '+name);continue
   x=hits[0]; score=x['metrics'][metric]; vals.append(score*100)
   runrows.append({'method':method,'task':task,'seed':int(re.search(r'_seed(\d+)',name).group(1)),'learning_rate':s['learning_rate'],'score':score,'metric':metric,'path':x['path'],'metrics':x['metrics']})
  chosen+=runrows
  if basename in byname:
   actual=byname[basename][0]['metrics'][metric]
   if abs(actual-s['validation_score'])>1e-9: issues.append('Manifest score mismatch '+basename)
  if vals and abs(max(vals)/100-s['validation_score'])>1e-9: issues.append('Best seed mismatch '+method+' '+task)
  # independently recompute 3-seed LR selection using same run family
  grids={}
  if task!='wnli':
   for lr in ['2em5','3em5','5em5']:
    scores=[]
    for seed in [42,43,44]:
     name=re.sub(r'_lr[^_]+_seed\d+_(grid|final)$',f'_lr{lr}_seed{seed}_grid',basename)
     if name in byname:scores.append(byname[name][0]['metrics'][metric])
    if len(scores)==3:grids[lr]=statistics.mean(scores)
   if len(grids)!=3:issues.append('Incomplete LR grid '+method+' '+task)
   elif grids[s['learning_rate'].replace('e-','em')]<max(grids.values())-1e-12:issues.append('LR mean mismatch '+method+' '+task)
  summary.append({'method':method,'task':task,'n':len(vals),'mean':statistics.mean(vals) if vals else None,'sd':statistics.stdev(vals) if len(vals)>1 else None,'learning_rate':s['learning_rate'],'metric':metric,'grid_means':grids})
(r/'summary.json').write_text(json.dumps({'summary':summary,'issues':issues,'runs':chosen},indent=2))
print('Records',len(records),'selected runs',len(chosen),'cells',len(summary));print('Issues:',issues)
tasks=['cola','sst2','mrpc','stsb','qqp','mnli','qnli','rte','wnli']; labels={'mha':'MHA','gqa':'GQA','collaborative':'Collaborative MHA','gt_mha_residual':'GT-MHA'}
lines=['# GLUE validation across fine-tuning seeds','','Scores are percentages. Mean ± sample standard deviation at the learning rate used for the official submission. Seeds 42–46 for each task except WNLI (seed 42 only). MRPC/QQP use mean accuracy and F1; STS-B uses mean Pearson and Spearman; MNLI uses the recorded matched accuracy. These are validation scores, not the official test scores.','','| Method | '+' | '.join(t.upper() for t in tasks)+' |','|---|'+'---|'*len(tasks)]
tex=['% Requires booktabs. Scores in percentage points; sample SD.','\\begin{tabular}{l'+'c'*len(tasks)+'}','\\toprule','Method & '+' & '.join(t.upper() for t in tasks)+r' \\','\\midrule']
for m,label in labels.items():
 cells=[];tc=[]
 for t in tasks:
  s=next(z for z in summary if z['method']==m and z['task']==t)
  cell=f"{s['mean']:.2f}" if s['mean'] is not None else 'MISSING'
  if s['sd'] is not None:cell+=f" ± {s['sd']:.2f}"
  cells.append(cell);tc.append('$'+cell.replace('±',r'\pm')+'$')
 lines.append('| '+label+' | '+' | '.join(cells)+' |');tex.append(label+' & '+' & '.join(tc)+r' \\')
lines+=['','## Interpretation and provenance','','The learning rate is selected using validation means over seeds 42–44; seeds 45–46 are additional runs at that rate. Consequently, these validation summaries reflect model selection and are not independent estimates. They characterize fine-tuning variability conditional on the selected pretrained encoder, not pretraining variability. No nine-task aggregate uncertainty is calculated.','','The official submission manifests and original metrics are retained alongside this report. summary.json records every included run path, metric, seed and learning rate. Historical runs and identity ablations are excluded from this four-method table.','','## Audit findings','']+[('- '+i) for i in issues]
if not issues:lines+=['All 36 cells have the expected run counts; selected-checkpoint scores and best-seed scores agree with the manifests; all 32 three-learning-rate grids reproduce the selected learning rate.']
tex += [r'\bottomrule',r'\end{tabular}']
(r/'validation_summary.md').write_text('\n'.join(lines)+'\n');(r/'validation_table.tex').write_text('\n'.join(tex)+'\n')
print('\n'.join(lines[:10]))
