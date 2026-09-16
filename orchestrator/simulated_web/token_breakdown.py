"""Offline estimates for visible reasoning and tool-argument token allocation."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from urllib.parse import urlsplit

try:
    from tokenizers import Tokenizer
except ImportError:
    Tokenizer = None


CATEGORIES = ('search_read', 'wiki_write', 'other_unknown')
SERIALIZATION = ('String arguments are counted verbatim; other JSON values use '
                 'json.dumps(ensure_ascii=False, separators=(",", ":"), sort_keys=True). '
                 'URL escapes are preserved; each field is tokenized separately without special tokens.')
LIMITATIONS = [
    'Estimates tokenize visible parsed fields only, not the original server token stream.',
    'Hidden or unlogged thinking, tool names, message syntax, prompts and chat templates are excluded.',
    'Missing reasoning fields mean unavailable, not evidence of zero reasoning.',
    'Native eval_count is reported separately; differences do not establish overhead or exact reconstruction.',
    'The user must verify tokenizer compatibility with the recorded model; a fingerprint does not prove compatibility.',
    'Missing argument fields have no countable text; missing_argument_calls flags incomplete visible totals.',
    'Failed assignments are included. Tool categories describe requests, not successful execution.',
    'Only assignments listed in results.json are included; an interrupted session may have unlisted work.',
    'Content is counted once from assistant events; final result answers are duplicate copies and are not recounted.',
]


def canonical(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), sort_keys=True, allow_nan=False)


def category(name, arguments):
    parsed = arguments
    if isinstance(parsed, str):
        try:
            parsed = json.loads(parsed)
        except ValueError:
            return 'other_unknown'
    if not isinstance(parsed, dict):
        return 'other_unknown'
    if name == 'search':
        return 'search_read'
    if name == 'click':
        return 'search_read'
    if name == 'open':
        url = parsed.get('url')
        if not isinstance(url, str):
            return 'other_unknown'
        try:
            parts = urlsplit(url)
        except ValueError:
            return 'other_unknown'
        if parts.scheme != 'https' or parts.netloc not in ('docs.test', 'wiki.test') or parts.fragment:
            return 'other_unknown'
        if parts.netloc == 'wiki.test' and parts.path == '/save':
            return 'wiki_write'
        return 'search_read'
    return 'other_unknown'


def summarize(counts):
    counts = dict(counts)
    denominator = counts['reasoning_tokens'] + counts['tool_argument_tokens']
    counts['denominator_tokens'] = denominator
    counts['shares'] = {
        key: counts[key] / denominator if denominator else None
        for key in ('reasoning_tokens', 'tool_argument_tokens',
                    *(name + '_tokens' for name in CATEGORIES))
    }
    counts['reasoning_availability'] = (
        'unavailable' if not counts['reasoning_present_events'] else
        'partial' if counts['reasoning_missing_events'] else 'available')
    return counts


def empty_counts():
    return Counter(dict.fromkeys((
        'assistant_events', 'reasoning_present_events', 'reasoning_missing_events',
        'reasoning_tokens', 'content_tokens', 'final_content_tokens', 'tool_argument_tokens',
        'missing_argument_calls', 'model_response_events', 'native_eval_count_events',
        'native_eval_count_missing_events', 'native_eval_count_tokens',
        *(name + '_tokens' for name in CATEGORIES)), 0))


def load_inputs(run_root):
    root = Path(run_root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError('Run root must be a directory')
    results = json.loads((root / 'results.json').read_text())
    settings = json.loads((root / 'settings.json').read_text())
    if not isinstance(results, list) or not isinstance(settings, dict):
        raise ValueError('Expected results.json list and settings.json object')
    loaded, seen = [], set()
    for row in results:
        if not isinstance(row, dict) or not isinstance(row.get('agent'), str):
            raise ValueError('Each assignment requires an agent string')
        relative = row.get('log_path')
        if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
            raise ValueError('Each assignment requires a relative log_path')
        path = (root / relative).resolve(strict=True)
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError('Assignment log_path must be a file inside run root')
        if path in seen:
            raise ValueError('Duplicate assignment log_path would double count tokens')
        seen.add(path)
        events = []
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            event = json.loads(line)
            if not isinstance(event, dict):
                raise ValueError(f'{relative}:{number}: expected event object')
            events.append(event)
        loaded.append((row, events))
    return root, settings, loaded


def load_tokenizer(source):
    path = Path(source).resolve(strict=True)
    if path.is_dir():
        path = (path / 'tokenizer.json').resolve(strict=True)
    if not path.is_file():
        raise ValueError('Tokenizer must be a local tokenizer.json file or directory containing one')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if Tokenizer is None:
        raise ValueError('The optional tokenizers package is required; no character-count fallback is used')
    try:
        tokenizer = Tokenizer.from_file(str(path))
        tokenizer.no_truncation()
        tokenizer.no_padding()
        tokenizer.encode('', add_special_tokens=False)
    except Exception as error:
        raise ValueError(f'Cannot load local tokenizer {path}: {error}') from error
    return tokenizer, {'path': str(path), 'sha256': digest, 'special_tokens': False,
                       'truncation': False, 'padding': False}


def build_report(run_root, tokenizer_path):
    root, settings, loaded = load_inputs(run_root)
    tokenizer, provenance = load_tokenizer(tokenizer_path)

    def tokens(value):
        if not isinstance(value, str):
            raise ValueError('Logged reasoning and content fields must be strings when present')
        return len(tokenizer.encode(value, add_special_tokens=False).ids)

    aggregate, agents, assignments = empty_counts(), {}, []
    for row, events in loaded:
        counts = empty_counts()
        for event in events:
            if event.get('event') == 'model_response':
                counts['model_response_events'] += 1
                metadata = event.get('metadata', {})
                if not isinstance(metadata, dict):
                    raise ValueError('model_response metadata must be an object')
                value = metadata.get('eval_count')
                if type(value) is int and value >= 0:
                    counts['native_eval_count_tokens'] += value
                    counts['native_eval_count_events'] += 1
                else:
                    counts['native_eval_count_missing_events'] += 1
            if event.get('event') != 'assistant':
                continue
            message = event.get('message')
            if not isinstance(message, dict):
                raise ValueError('Assistant event requires a message object')
            counts['assistant_events'] += 1
            # Ollama calls its visible reasoning field "thinking". Also accept
            # "reasoning" from locally converted logs, without counting both.
            keys = [key for key in ('thinking', 'reasoning') if key in message and message[key] is not None]
            if len(keys) > 1:
                raise ValueError('Ambiguous assistant message contains both thinking and reasoning')
            counts['reasoning_present_events' if keys else 'reasoning_missing_events'] += 1
            if keys:
                counts['reasoning_tokens'] += tokens(message[keys[0]])
            content = tokens(message.get('content', ''))
            counts['content_tokens'] += content
            calls = message.get('tool_calls', [])
            if not isinstance(calls, list):
                raise ValueError('tool_calls must be a list')
            if not calls and event.get('assignment_phase', event.get('phase')) != 'preparation':
                counts['final_content_tokens'] += content
            for call in calls:
                function = call.get('function', {}) if isinstance(call, dict) else {}
                if not isinstance(function, dict) or 'arguments' not in function:
                    counts['missing_argument_calls'] += 1
                    continue
                arguments = function['arguments']
                count = tokens(arguments if isinstance(arguments, str) else canonical(arguments))
                counts['tool_argument_tokens'] += count
                counts[category(function.get('name'), arguments) + '_tokens'] += count
        aggregate.update(counts)
        agents.setdefault(row['agent'], empty_counts()).update(counts)
        assignments.append({**{key: row.get(key) for key in ('agent', 'id', 'slot', 'status', 'log_path')},
                            **summarize(counts)})
    return {
        'schema_version': 1, 'run_root': str(root), 'tokenizer': provenance,
        'model_identity_from_settings': {key: value for key, value in settings.items()
                                         if 'model' in key or key == 'settings'},
        'serialization': SERIALIZATION,
        'denominator': 'visible reasoning tokens + all visible tool argument tokens; content excluded',
        'content_definition': 'All assistant content; no-tool-call content outside preparation also shown as final_content_tokens.',
        'native_count_definition': 'Sum of valid eval_count in model_response events, including responses without assistant events.',
        'limitations': LIMITATIONS, 'assignment_count': len(assignments),
        'aggregate': summarize(aggregate),
        'agents': {name: summarize(counts) for name, counts in sorted(agents.items())},
        'assignments': assignments,
    }


def markdown(report):
    def cell(value):
        return str(value).replace('|', '\\|').replace('\n', ' ').replace('\r', ' ')

    def row(label, counts):
        shares = counts['shares']
        share = lambda key: 'n/a' if shares[key] is None else f'{shares[key]:.2%}'
        return '| ' + ' | '.join(map(cell, (label, counts['denominator_tokens'],
            counts['reasoning_tokens'], share('reasoning_tokens'),
            counts['tool_argument_tokens'], share('tool_argument_tokens'),
            counts['search_read_tokens'], share('search_read_tokens'),
            counts['wiki_write_tokens'], share('wiki_write_tokens'),
            counts['other_unknown_tokens'], counts['content_tokens'],
            counts['native_eval_count_tokens'], counts['reasoning_availability']))) + ' |'

    lines = ['# Visible token allocation estimate', '', report['denominator'], '',
             '| Scope | Denominator | Reasoning | Share | All tool arguments | Share | Search/read | Share | Wiki write | Share | Other/unknown | Content (excluded) | Native eval_count | Reasoning availability |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|',
             row('All assignments', report['aggregate'])]
    lines.extend(row(agent, counts) for agent, counts in report['agents'].items())
    lines.extend(row(f"{a['agent']} / slot {a['slot']} / {a['id']} ({a['status']})", a)
                 for a in report['assignments'])
    lines.extend(['', 'Tokenizer: ' + cell(report['tokenizer']['path']),
                  '', 'SHA-256: ' + report['tokenizer']['sha256'], '',
                  'Model settings: ' + cell(canonical(report['model_identity_from_settings'])), '',
                  SERIALIZATION, '', report['native_count_definition'], '',
                  'Completeness counters (aggregate): ' + canonical({key: value for key, value in report['aggregate'].items()
                      if key.endswith('_events') or key == 'missing_argument_calls'}), '',
                  *('- ' + item for item in LIMITATIONS), ''])
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, required=True,
                        help='Local tokenizer.json file or local directory containing tokenizer.json')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args(argv)
    output = args.output_dir or args.run_root / 'token-breakdown'
    try:
        if output.exists() or output.is_symlink():
            raise ValueError('Output directory must be fresh; existing artifacts are never overwritten')
        report = build_report(args.run_root, args.tokenizer)
        # Finish parsing, validation and serialization before creating any artifact.
        encoded, rendered = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n', markdown(report)
        output.mkdir(parents=True, exist_ok=False)
        (output / 'report.json').write_text(encoded)
        (output / 'report.md').write_text(rendered)
    except (OSError, ValueError, TypeError) as error:
        parser.exit(2, f'token-breakdown: {error}\n')
    print(f'Wrote {output / "report.json"} and {output / "report.md"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
