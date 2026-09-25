"""Extract development-only prefixes; prints JSON, makes no remote changes."""
import hashlib,json,random,sys
from pathlib import Path
p=Path(sys.argv[1]); raw=p.read_bytes(); text=raw.decode('utf-8'); stories=text.split('<|endoftext|>')
eligible=[]; excluded=0
for i,s in enumerate(stories):
    cut=next((j for j in range(120,min(161,len(s))) if s[j].isspace()),None)
    if len(s)<256 or cut is None:
        excluded+=1; continue
    eligible.append((len(s),i,cut))
eligible.sort(); rng=random.Random(20260919); rows=[]
for b in range(4):
    pool=eligible[len(eligible)*b//4:len(eligible)*(b+1)//4]
    for length,i,cut in rng.sample(pool,5):
        s=stories[i]
        rows.append({'prompt_id':f'dev_{len(rows)+1:03d}','story_id':i,'story_sha256':hashlib.sha256(s.encode()).hexdigest(),'story_characters':length,'length_quartile':b+1,'prefix':s[:cut],'prefix_characters':cut,'reference_continuation':s[cut:],'use':'engineering_only_exclude_from_final_evaluation'})
assert len({r['story_id'] for r in rows})==20
assert all(120<=r['prefix_characters']<=160 for r in rows)
print(json.dumps({'source_path':str(p),'source_sha256':hashlib.sha256(raw).hexdigest(),'selection_seed':20260919,'eligible_stories':len(eligible),'excluded_stories':excluded,'training_overlap_check':'not yet performed; development set only','prompts':rows},ensure_ascii=False,indent=2))
