"""Export recorded MuSiQue answers, without inference or gold-based extraction."""
import argparse
from collections import Counter
import fcntl
import hashlib
import json
from pathlib import Path


def write_jsonl(path, rows):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
    temporary.replace(path)


def export_records(records, locations, agents, output):
    predictions = {f'agent-{i}': [] for i in range(1, agents + 1)}
    statuses = []
    for record in records:
        folder = locations[record['id']]
        outcome = json.loads((folder / 'outcome.json').read_text())
        if outcome['id'] != record['id']:
            raise ValueError('Outcome ID mismatch')
        # Derive the path locally rather than trusting an absolute recorded path.
        attempt = outcome['attempt']
        if not isinstance(attempt, int) or attempt < 1:
            raise ValueError('Invalid attempt number')
        result_path = folder / f'attempt-{attempt:03d}' / 'results.json'
        results = json.loads(result_path.read_text()) if result_path.exists() else []
        by_agent = {result['agent']: result for result in results}
        if len(by_agent) != len(results) or set(by_agent) - set(predictions):
            raise ValueError('Duplicate or unexpected result agent')
        for agent, rows in predictions.items():
            result = by_agent.get(agent, {})
            status = result.get('status', 'missing_results')
            answer = result.get('answer', '')
            if not isinstance(answer, str):
                raise ValueError('Answer must be a string')
            rows.append({'id': record['id'],
                         'predicted_answer': answer if status == 'complete' else '',
                         'predicted_support_idxs': [], 'predicted_answerable': True})
            statuses.append({'id': record['id'], 'agent': agent, 'status': status,
                             'returncode': outcome['returncode'], 'results_path': str(result_path),
                             'error': result.get('error')})
    output.mkdir(exist_ok=True)
    for agent, rows in predictions.items():
        write_jsonl(output / f'{agent}.predictions.jsonl', rows)
    write_jsonl(output / 'gold.jsonl', records)
    write_jsonl(output / 'statuses.jsonl', statuses)
    metadata = {
        'questions': len(records), 'agents': agents,
        'answer_policy': 'Verbatim final response for complete status; empty for all other statuses. No extraction.',
        'support_policy': 'No support prediction implemented; empty lists are placeholders, not evidence scores.',
        'answerability_policy': 'Always true: this harness accepts MuSiQue-Ans only.',
        'status_counts': dict(Counter(row['status'] for row in statuses)),
        'files_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in sorted(output.glob('*.jsonl'))}}
    temporary = output / 'export.json.tmp'
    temporary.write_text(json.dumps(metadata, indent=2) + '\n')
    temporary.replace(output / 'export.json')
    print(f'Exported {len(records)} questions x {agents} agents to {output}', flush=True)


def export_run(dataset, run_root, shard=None):
    raw = dataset.read_bytes()
    records = [json.loads(line) for line in raw.splitlines() if line.strip()]
    ids = [r['id'] for r in records]
    if not ids or len(ids) != len(set(ids)) or any(r.get('answerable') is False for r in records):
        raise ValueError('Expected nonempty MuSiQue-Ans dataset with unique IDs')
    # Each finished shard calls this. A lock ensures the last finisher can merge
    # all results without concurrent writers; no waiting for unfinished jobs.
    with (run_root / '.prediction-export.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifests = {}
        locations = {}
        for path in sorted(run_root.glob('shard-*/manifest.json')):
            manifest = json.loads(path.read_text())
            number = manifest['shard']
            if path.parent.name != f'shard-{number}' or number in manifests:
                raise ValueError('Invalid shard manifest location')
            if manifest['dataset_sha256'] != hashlib.sha256(raw).hexdigest():
                raise ValueError('Dataset differs from the recorded run')
            manifests[number] = manifest
            for qid in manifest['ids']:
                if qid not in ids or qid in locations:
                    raise ValueError('Unexpected or duplicate assigned question')
                locations[qid] = path.parent / hashlib.sha256(qid.encode()).hexdigest()
        if not manifests:
            raise ValueError('No shard manifests found')
        first = next(iter(manifests.values()))
        keys = ('shards', 'agents', 'model', 'model_digest', 'seed', 'context_length',
                'max_output_tokens', 'steps', 'timeout', 'source_hashes')
        if any(any(m.get(k) != first.get(k) for k in keys) for m in manifests.values()):
            raise ValueError('Shard experiment settings or model/source hashes differ')
        if shard is not None:
            if shard not in manifests:
                raise ValueError('Requested shard has no manifest')
            wanted = set(manifests[shard]['ids'])
            if any(not (locations[qid] / 'outcome.json').exists() for qid in wanted):
                raise ValueError('Requested shard is unfinished; no complete export available')
            export_records([r for r in records if r['id'] in wanted], locations,
                           first['agents'], run_root / f'shard-{shard}' / 'predictions')
        ready = (set(manifests) == set(range(first['shards'])) and set(locations) == set(ids)
                 and all((folder / 'outcome.json').exists() for folder in locations.values()))
        if ready:
            export_records(records, locations, first['agents'], run_root / 'predictions')
        elif shard is None:
            raise ValueError('Run is unfinished; refusing an incomplete combined export')
        else:
            print('Combined export deferred until every shard has terminal outcomes.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--shard', type=int)
    args = parser.parse_args()
    export_run(args.dataset, args.run_root, args.shard)


if __name__ == '__main__':
    main()
