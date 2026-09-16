"""User-launched MuSiQue-Ans shards; no labels or decompositions enter browser pages."""
import argparse
import hashlib
import http.client
import json
from pathlib import Path
import subprocess
import sys

from orchestrator.simulated_web.browser import MAX_TEXT


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def convert(record):
    """Keep every supplied paragraph, splitting long text without dropping characters."""
    pages = []
    links = []
    for position, paragraph in enumerate(record['paragraphs']):
        title, text = paragraph['title'], paragraph['paragraph_text']
        if not isinstance(title, str) or not isinstance(text, str):
            raise ValueError('Paragraph titles and text must be strings')
        chunks = [text[i:i + MAX_TEXT] for i in range(0, len(text), MAX_TEXT)] or ['']
        urls = [f'https://docs.test/p/{position}/{i}' for i in range(len(chunks))]
        links.append({'label': title[:200], 'url': urls[0]})
        for i, chunk in enumerate(chunks):
            navigation = [{'label': 'Collection index', 'url': 'https://docs.test/'}]
            if i + 1 < len(chunks):
                navigation.append({'label': 'Next part', 'url': urls[i + 1]})
            pages.append({'url': urls[i], 'title': title[:180] + f' (part {i + 1})',
                          'text': chunk, 'links': navigation})
    # Paginated indexes preserve reachability with the browser's 20-link limit.
    groups = [links[i:i + 18] for i in range(0, len(links), 18)]
    for i, group in enumerate(groups):
        navigation = list(group)
        navigation.append({'label': 'Field notebook', 'url': 'https://wiki.test/'})
        if i + 1 < len(groups):
            navigation.append({'label': 'More documents', 'url': f'https://docs.test/index/{i + 1}'})
        pages.append({'url': 'https://docs.test/' if i == 0 else f'https://docs.test/index/{i}',
                      'title': 'Document collection',
                      'text': 'Available documents:\n' + '\n'.join(link['label'] for link in group),
                      'links': navigation})
    if len(pages) > 10000:
        raise ValueError('Corpus exceeds browser page limit')
    return pages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--shard', type=int, required=True)
    parser.add_argument('--shards', type=int, default=4)
    parser.add_argument('--model', default='qwen3.5:2b')
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--agents', type=int, default=1)
    parser.add_argument('--context-length', type=int, default=16384)
    parser.add_argument('--max-output-tokens', type=int, default=2048)
    parser.add_argument('--steps', type=int, default=24)
    parser.add_argument('--timeout', type=int, default=600)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    if not (0 <= args.shard < args.shards and 1 <= args.agents <= 32 and
            1 <= args.max_output_tokens <= 32768 and 1 <= args.steps <= 500 and 1 <= args.timeout <= 3600 and
            1024 <= args.context_length <= 32768 and 1 <= args.port <= 65535):
        parser.error('Invalid shard, agent count, step budget, timeout, context or port')
    raw = args.dataset.read_bytes()
    records = [json.loads(line) for line in raw.splitlines() if line.strip()]
    ids = [record['id'] for record in records]
    if not records or len(set(ids)) != len(ids):
        raise ValueError('Dataset must be nonempty with unique IDs')
    # Validate all records before selecting any; never silently drop malformed tasks.
    for record in records:
        if not isinstance(record['id'], str) or record.get('answerable') is False:
            raise ValueError('Use official MuSiQue-Ans, not unanswerable Full records')
        if not isinstance(record['question'], str) or not 1 <= len(record['question']) <= 8000:
            raise ValueError('Missing or oversized question')
        if not record['paragraphs']:
            raise ValueError('Missing paragraphs')
        convert(record)
    # Hop count is encoded in official IDs. Sorting within hop groups by hash
    # avoids file-order bias; round-robin assignment balances each hop group.
    records.sort(key=lambda r: (r['id'].split('__')[0], hashlib.sha256(r['id'].encode()).hexdigest()))
    assigned = records[args.shard::args.shards]
    connection = http.client.HTTPConnection('127.0.0.1', args.port, timeout=10)
    try:
        connection.request('GET', '/api/tags')
        response = connection.getresponse()
        if response.status != 200:
            raise ValueError('Cannot read local Ollama model metadata')
        models = json.loads(response.read())['models']
    finally:
        connection.close()
    model = next((m for m in models if m['name'] == args.model), None)
    if model is None:
        raise ValueError(f'Model {args.model} must be downloaded before submission')
    configuration = {key: value for key, value in vars(args).items() if key not in ('dataset', 'output', 'port')}
    configuration.update(dataset_sha256=hashlib.sha256(raw).hexdigest(), model_digest=model['digest'],
                         model_details=model['details'], ids=[r['id'] for r in assigned],
                         source_hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                        for p in Path(__file__).parent.glob('*.py')})
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output / 'manifest.json'
    if manifest.exists():
        if json.loads(manifest.read_text()) != configuration:
            raise ValueError('Resume configuration, source, model or dataset mismatch; use a fresh RUN_ROOT')
    else:
        write_json(manifest, configuration)
    failures = 0
    for index, record in enumerate(assigned, 1):
        key = hashlib.sha256(record['id'].encode()).hexdigest()
        folder = args.output / key
        folder.mkdir(exist_ok=True)
        outcome_path = folder / 'outcome.json'
        if outcome_path.exists():
            outcome = json.loads(outcome_path.read_text())
            failures += outcome['returncode'] != 0
            print(f'[{index}/{len(assigned)}] recorded: {record["id"]}', flush=True)
            continue
        write_json(folder / 'source.json', record)  # Private gold/decomposition, never browser-visible.
        write_json(folder / 'pages.json', convert(record))
        write_json(folder / 'task.json', {'question': record['question']})
        attempt = 1
        while (folder / f'attempt-{attempt:03d}').exists():
            attempt += 1
        run_dir = folder / f'attempt-{attempt:03d}'
        command = [sys.executable, '-m', 'orchestrator.simulated_web.runner',
                   '--corpus', str(folder / 'pages.json'), '--task', str(folder / 'task.json'),
                   '--run-dir', str(run_dir), '--model', args.model, '--port', str(args.port),
                   '--agents', str(args.agents), '--context-length', str(args.context_length),
                   '--max-output-tokens', str(args.max_output_tokens),
                   '--steps', str(args.steps), '--timeout', str(args.timeout), '--seed', str(args.seed)]
        print(f'[{index}/{len(assigned)}] starting: {record["id"]}', flush=True)
        with (folder / f'attempt-{attempt:03d}.stdout').open('x') as stdout:
            # The child handles model socket timeouts. Slurm bounds the complete shard.
            result = subprocess.run(command, stdout=stdout, stderr=subprocess.STDOUT, check=False)
        outcome = {'id': record['id'], 'returncode': result.returncode,
                   'attempt': attempt, 'run_dir': str(run_dir)}
        write_json(outcome_path, outcome)
        failures += result.returncode != 0
        print(f'[{index}/{len(assigned)}] finished: {record["id"]}, exit={result.returncode}', flush=True)
    print(f'Shard finished: {len(assigned)} questions, {failures} nonzero exits. Answers are not graded.', flush=True)
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
