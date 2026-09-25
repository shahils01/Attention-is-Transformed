"""Auditable, resumable RCD judging. Standard library only; secrets stay off disk here."""
import argparse
import collections
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parent
MODEL = 'gpt-4.1-2025-04-14'
URL = 'https://llm.rcd.clemson.edu/v1/chat/completions'
SCORES = ('grammar', 'creativity', 'consistency', 'plot')
SCHEMA = {'type': 'object', 'additionalProperties': False, 'properties': {
    **{k: {'type': 'integer', 'minimum': 1, 'maximum': 10} for k in SCORES},
    'closure': {'type': 'string', 'enum': ['closed', 'open', 'unclear', 'not_assessed']},
    'assessment': {'type': 'string'}}, 'required': [*SCORES, 'closure', 'assessment']}

def encode(x):
    return json.dumps(x, ensure_ascii=False, sort_keys=True).encode()

def digest(x):
    return hashlib.sha256(x).hexdigest()

def save(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_bytes(encode(obj) + b'\n')
    tmp.replace(path)

def rows(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]

def prepare(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    system = Path(args.rubric).read_text().split('## System message\n', 1)[1].split('\n## User payload schema', 1)[0].strip()
    source = [r for p in args.input for r in rows(Path(p))]
    candidates, mapping, seen = [], [], set()
    for r in source:
        identity = [r['checkpoint_sha256'], r['prompt_id'], r['sample_index'], args.mode]
        cid = digest(encode(identity))
        if cid in seen:
            raise ValueError('Duplicate candidate')
        seen.add(cid)
        text = r['completion']
        cut = args.mode == 'excerpt' and len(text) > 320
        payload = {'mode': args.mode, 'prefix': r['prompt'],
                   'continuation': text[:320] if args.mode == 'excerpt' else text,
                   'evaluator_cut': cut}
        candidates.append({'id': cid, 'payload': payload})
        mapping.append({'id': cid, **{k: r[k] for k in ('model', 'checkpoint_sha256', 'prompt_id', 'sample_index', 'finish_reason')}})
    random.Random(20260919).shuffle(candidates)
    config = {'phase': args.phase, 'model': MODEL, 'url': URL, 'temperature': 0,
              'max_tokens': 512, 'schema': SCHEMA, 'system': system,
              'rubric_sha256': digest(Path(args.rubric).read_bytes()),
              'input_sha256': {p: digest(Path(p).read_bytes()) for p in args.input},
              'candidate_count': len(candidates), 'mode': args.mode,
              'runner_sha256': digest(Path(__file__).read_bytes()),
              'candidates_sha256': digest(encode(candidates))}
    save(out / 'config.json', config)
    save(out / 'candidates.json', candidates)
    save(out / 'private_mapping.json', mapping)
    print(json.dumps({'prepared': len(candidates), 'phase': args.phase, 'config_sha256': digest(encode(config))}))

def validate(raw, payload):
    if raw.get('model') != MODEL:
        raise ValueError('Unexpected returned model')
    choice = raw['choices'][0]
    if choice['finish_reason'] != 'stop' or choice['message'].get('refusal'):
        raise ValueError('Refusal or incomplete judge output')
    result = json.loads(choice['message']['content'])
    if set(result) != set(SCHEMA['required']):
        raise ValueError('Wrong fields')
    if any(type(result[k]) is not int or not 1 <= result[k] <= 10 for k in SCORES):
        raise ValueError('Invalid scores')
    if result['closure'] not in SCHEMA['properties']['closure']['enum'] or not isinstance(result['assessment'], str) or not result['assessment'].strip():
        raise ValueError('Invalid closure or assessment')
    if (result['closure'] == 'not_assessed') != bool(payload['evaluator_cut']):
        raise ValueError('Closure inconsistent with excerpt cut')
    return result

def run(args):
    out = Path(args.directory)
    lock = (out / '.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config = json.loads((out / 'config.json').read_text())
    candidates = json.loads((out / 'candidates.json').read_text())
    assert digest(encode(candidates)) == config['candidates_sha256']
    assert digest(Path(__file__).read_bytes()) == config['runner_sha256'], 'Runner changed; prepare a new run'
    assert config['model'] == MODEL and config['url'] == URL
    if config['phase'] == 'main':
        approval = json.loads((out / 'frozen.json').read_text())
        assert approval['config_sha256'] == digest(encode(config))
    key_path = Path.home() / '.config/tinystories-eval/rcd-token'
    assert key_path.stat().st_mode & 0o077 == 0, 'Token file must be private'
    token = key_path.read_text().strip()
    attempts = out / 'attempts'
    attempts.mkdir(exist_ok=True)
    # Reserve a conservative byte-based token cost BEFORE every attempt, including
    # unknown-outcome failures. Persisted reservations prevent resume overspending.
    reserved = sum(json.loads(p.read_text())['reserved_usd'] for p in attempts.glob('*.request.json'))
    processed = 0
    for candidate in candidates:
        cid = candidate['id']
        result_file = out / (cid + '.result.json')
        if result_file.exists():
            continue
        for schema_try in range(2):
            messages = [{'role': 'system', 'content': config['system']},
                        {'role': 'user', 'content': json.dumps(candidate['payload'], ensure_ascii=False)}]
            if schema_try:
                messages[0]['content'] += '\nReturn exactly the required JSON schema with all fields; no extra text.'
            body = {'model': MODEL, 'temperature': config['temperature'], 'max_tokens': config['max_tokens'],
                    'messages': messages, 'response_format': {'type': 'json_schema',
                    'json_schema': {'name': 'story_quality', 'strict': True, 'schema': config['schema']}}}
            base = attempts / (cid + '.' + str(schema_try))
            raw = None
            for transport_try in range(4):
                prefix = Path(str(base) + '.' + str(transport_try))
                req_file = Path(str(prefix) + '.request.json')
                response_file = Path(str(prefix) + '.response.json')
                error_file = Path(str(prefix) + '.error.json')
                if response_file.exists():
                    raw = json.loads(response_file.read_text())
                    break
                if req_file.exists():
                    if not error_file.exists():
                        raise RuntimeError('Uncertain previous request; inspect before resuming: ' + str(prefix))
                    error = json.loads(error_file.read_text())
                    if not error['retryable']:
                        raise RuntimeError('Non-retryable API failure; inspect error artifact')
                    continue
                reserve = ((len(encode(body)) + 1024) * 2 + config['max_tokens'] * 8) / 1e6
                if reserved + reserve > args.budget_usd:
                    raise RuntimeError('Conservative run budget reached')
                save(req_file, {'body': body, 'reserved_usd': reserve, 'time': time.time()})
                reserved += reserve
                request = urllib.request.Request(URL, data=encode(body), headers={
                    'Authorization': 'Bearer ' + token, 'Content-Type': 'application/json'})
                try:
                    with urllib.request.urlopen(request, timeout=180) as response:
                        raw = json.load(response)
                    save(response_file, raw)
                    break
                except urllib.error.HTTPError as exc:
                    retryable = exc.code == 429 or 500 <= exc.code < 600
                    save(error_file, {'status': exc.code, 'retryable': retryable, 'time': time.time()})
                    if not retryable:
                        raise RuntimeError('API HTTP status ' + str(exc.code)) from None
                    retry_after = exc.headers.get('Retry-After', '')
                    delay = max(2 ** (transport_try + 1), float(retry_after) if retry_after.replace('.', '', 1).isdigit() else 0)
                    time.sleep(delay)
                except (urllib.error.URLError, TimeoutError):
                    # Outcome may have been billed; no automatic duplicate request.
                    raise RuntimeError('Network outcome uncertain; request preserved for inspection') from None
            if raw is None:
                raise RuntimeError('Transport retries exhausted')
            try:
                result = validate(raw, candidate['payload'])
                save(result_file, {'id': cid, 'status': 'valid', 'scores': result,
                                   'schema_attempt': schema_try, 'response_path': str(response_file)})
                break
            except (ValueError, KeyError, TypeError, IndexError) as exc:
                if schema_try == 1:
                    save(result_file, {'id': cid, 'status': 'missing', 'reason': str(exc)})
        processed += 1
        if processed % 10 == 0:
            print(json.dumps({'new_results': processed}), flush=True)
        if processed >= args.limit:
            break
    summary(out)

def summary(out):
    results = [json.loads(p.read_text()) for p in out.glob('*.result.json')]
    usage = collections.Counter()
    for p in (out / 'attempts').glob('*.response.json'):
        u = json.loads(p.read_text()).get('usage', {})
        usage.update({k: u.get(k, 0) for k in ('prompt_tokens', 'completion_tokens')})
        usage['cached_tokens'] += u.get('prompt_tokens_details', {}).get('cached_tokens', 0)
    cost = (2 * usage['prompt_tokens'] - 1.5 * usage['cached_tokens'] + 8 * usage['completion_tokens']) / 1e6
    report = {'results': len(results), 'status_counts': dict(collections.Counter(r['status'] for r in results)),
              'usage': dict(usage), 'estimated_usd_from_recorded_usage': cost,
              'unknown_or_failed_attempts': len(list((out/'attempts').glob('*.request.json'))) - len(list((out/'attempts').glob('*.response.json')))}
    save(out / 'summary.json', report)
    print(json.dumps(report), flush=True)

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--input', nargs='+', required=True)
    p.add_argument('--rubric', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--phase', choices=['development', 'main'], required=True)
    p.add_argument('--mode', choices=['full_story', 'excerpt'], default='full_story')
    p = sub.add_parser('run')
    p.add_argument('directory')
    p.add_argument('--budget-usd', type=float, required=True)
    p.add_argument('--limit', type=int, default=160)
    p = sub.add_parser('summary')
    p.add_argument('directory')
    args = parser.parse_args()
    if args.command == 'prepare': prepare(args)
    elif args.command == 'run':
        assert args.budget_usd > 0 and args.limit > 0
        run(args)
    else: summary(Path(args.directory))

if __name__ == '__main__':
    main()
