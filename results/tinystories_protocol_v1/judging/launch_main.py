"""Prepare the complete main set only after generation and development finish."""
import argparse
import collections
import json
from pathlib import Path
import subprocess
import sys
import judge

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent

def main():
    dev = ROOT / 'development_v1'
    config = json.loads((dev / 'config.json').read_text())
    candidates = json.loads((dev / 'candidates.json').read_text())
    assert len(candidates) == 160
    assert judge.digest(judge.encode(candidates)) == config['candidates_sha256']
    assert config['runner_sha256'] == judge.digest((ROOT / 'judge.py').read_bytes())
    assert config['rubric_sha256'] == judge.digest((BASE / 'judge_rubric.md').read_bytes())
    for c in candidates:
        r = json.loads((dev / (c['id'] + '.result.json')).read_text())
        assert r['status'] == 'valid', 'Development invalid results require review'
        response = Path(r['response_path'])
        if not response.is_absolute():
            response = BASE / response
        judge.validate(json.loads(response.read_text()), c['payload'])
    inputs, keys, checkpoints = [], None, set()
    prompt_text = {}
    cap_hits = 0
    for i in range(8):
        folder = BASE / 'main_run' / 'outputs' / f'{i:02d}'
        assert json.loads((folder / 'complete.json').read_text())['status'] == 'generation_complete'
        path = folder / 'generations.jsonl'
        records = judge.rows(path)
        assert len(records) == 1000
        identity = {(r['prompt_id'], r['sample_index']) for r in records}
        assert len(identity) == 1000
        assert len({r['prompt_id'] for r in records}) == 200
        assert set(r['sample_index'] for r in records) == set(range(5))
        assert keys is None or keys == identity
        keys = identity
        shas = {r['checkpoint_sha256'] for r in records}
        assert len(shas) == 1
        checkpoints.update(shas)
        for r in records:
            assert r['cap'] == 3000 and r['context_length'] == 512
            assert r['temperature'] == 1.0 and r['top_k'] == 20
            assert r['finish_reason'] in ('stop_sequence', 'max_new_tokens')
            prior = prompt_text.setdefault(r['prompt_id'], (r['prompt'], {}))
            assert prior[0] == r['prompt']
            old_seed = prior[1].setdefault(r['sample_index'], r['seed'])
            assert old_seed == r['seed']
            cap_hits += r['finish_reason'] == 'max_new_tokens'
        inputs.append(str(path))
    assert len(checkpoints) == 8
    out = ROOT / 'main_full_v1'
    if not out.exists():
        judge.prepare(argparse.Namespace(output=str(out), rubric=str(BASE / 'judge_rubric.md'),
            input=inputs, mode='full_story', phase='main'))
        new = json.loads((out / 'config.json').read_text())
        assert all(new[k] == config[k] for k in ('model','url','system','schema','temperature','max_tokens','rubric_sha256','runner_sha256'))
        judge.save(out / 'frozen.json', {
            'config_sha256': judge.digest(judge.encode(new)),
            'development_config_sha256': judge.digest(judge.encode(config)),
            'development_schema_valid': 160,
            'main_candidates': 8000, 'cap_hits_retained': cap_hits,
            'status': 'frozen_for_primary_scoring; reliability_audit_pending',
            'calibration_limitations': 'Four development examples reviewed blinded by assistant; one missed explicit fast/not-fast contradiction. No human agreement or repeatability study completed. Scores remain provisional pending reliability audit. No model rankings used to select judge or change rubric.'})
    else:
        existing = json.loads((out / 'config.json').read_text())
        assert existing['input_sha256'] == {p:judge.digest(Path(p).read_bytes()) for p in inputs}
    subprocess.run([sys.executable, '-u', str(ROOT / 'judge.py'), 'run', str(out),
                    '--budget-usd', '200', '--limit', '8000'], check=True)

if __name__ == '__main__':
    main()
