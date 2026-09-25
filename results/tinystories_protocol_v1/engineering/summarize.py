"""Summarize engineering outputs without ranking story quality."""
import json,statistics
from pathlib import Path
root=Path(__file__).resolve().parent
manifest=json.loads((root/'checkpoint_manifest.json').read_text());result=[]
for i,item in enumerate(manifest['checkpoints']):
 out=root/'outputs'/f'{i:02d}'
 rows=[json.loads(l) for l in (out/'generations.jsonl').read_text().splitlines()] if (out/'generations.jsonl').exists() else []
 assert len({r['prompt_id'] for r in rows})==len(rows)
 for r in rows:
  assert len(r['raw_completion'])==len(r['generated_token_ids'])==r['generated_tokens_including_stop']
  assert (r['finish_reason']=='stop_sequence')==r['raw_completion'].endswith('<|endoftext|>')
  assert r['finish_reason']=='stop_sequence' or r['generated_tokens_including_stop']==r['cap']
 checks=json.loads((out/'parity.json').read_text()) if (out/'parity.json').exists() else []
 meta=json.loads((out/'metadata.json').read_text()) if (out/'metadata.json').exists() else {}
 result.append({'model':item['model'],'status':'complete' if (out/'complete.json').exists() else 'incomplete','samples':len(rows),'parity_passed':all(c['passed'] for c in checks) if checks else None,'failed_checks':[c for c in checks if not c['passed']],'checkpoint_step':meta.get('checkpoint_step'),'gpu':meta.get('gpu'),'normal_stop':sum(r['finish_reason']=='stop_sequence' for r in rows),'cap_hits':sum(r['finish_reason']=='max_new_tokens' for r in rows),'mean_completion_characters':statistics.mean(len(r['completion']) for r in rows) if rows else None,'window_shifted':sum(r['first_shift_before_generated_token_zero_based'] is not None for r in rows),'generation_seconds':sum(r['elapsed_seconds'] for r in rows)})
print(json.dumps(result,indent=2))
