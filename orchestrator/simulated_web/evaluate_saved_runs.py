"""Read-only, standard-library saved 9d mechanical evidence extraction."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys


SCHEMA = 'saved-9d-evaluation-v1'
SEMANTIC_FIELDS = ('recognition', 'directed_request', 'reply_relevance', 'uptake', 'error_propagation')


def objects(value, location):
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise ValueError(f'{location}: expected a list of objects')
    return value


class Artifacts:
    """Never follow references outside the supplied run, including symlinks."""
    def __init__(self, root):
        self.root = root
        self.inventory = {}
        self.missing = []
        self.line_numbers = {}

    def read(self, name, required=False, lines=False):
        path = self.root / name
        if not path.resolve().is_relative_to(self.root):
            raise ValueError(f'Artifact escapes run directory: {name}')
        if not path.exists():
            if required:
                raise ValueError(f'Missing required artifact: {path}')
            self.missing.append(str(name))
            return None
        raw = path.read_bytes()
        self.inventory[str(name)] = hashlib.sha256(raw).hexdigest()
        try:
            if lines:
                pairs = [(i, line) for i, line in enumerate(raw.decode().splitlines(), 1) if line.strip()]
                self.line_numbers[str(name)] = [i for i, _ in pairs]
                return [json.loads(line) for _, line in pairs]
            return json.loads(raw)
        except (ValueError, UnicodeError) as error:
            raise ValueError(f'Malformed JSON: {path}: {error}') from error


def nonnegative(value):
    return type(value) is int and value >= 0


def tally(rows, field):
    known = [r[field] for r in rows if nonnegative(r.get(field))]
    return {'observed_sum': sum(known) if known else None,
            'known_rows': len(known), 'total_rows': len(rows),
            'complete': bool(rows) and len(known) == len(rows)}


def relation(author, agent, settings):
    labels = settings.get('visible_labels', {})
    if author is None or agent not in labels:
        return 'unknown'
    if author in (agent, labels[agent]):
        return 'own'
    if author in labels or author in labels.values():
        return 'peer'
    return 'unknown'


def destination_relation(destination, agent, settings):
    notebooks = settings.get('notebooks', {})
    owners = [owner for owner, slug in notebooks.items()
              if destination in (slug, 'https://wiki.test/page/' + slug)]
    return ('own' if owners[0] == agent else 'cross') if len(owners) == 1 and agent in notebooks else 'unknown'


def history_observations(messages, agent, number, source, settings):
    """Explicit tool call origin and nonempty response body; no semantic label."""
    objects(messages, f'{source}/{agent}')
    model_indexes = [i for i, m in enumerate(messages) if m.get('role') == 'assistant'
                     and not str(m.get('content', '')).startswith('[Host-')]
    pending = {}
    observed = []
    gaps = Counter()
    for index, message in enumerate(messages):
        if message.get('role') == 'assistant':
            for call in message.get('tool_calls', []):
                if not isinstance(call, dict) or not isinstance(call.get('function'), dict):
                    raise ValueError(f'{source}: malformed tool call')
                pending[call.get('id')] = (index, call['function'],
                    'host' if str(message.get('content', '')).startswith('[Host-') else 'model')
        if message.get('role') != 'tool':
            continue
        pair = pending.pop(message.get('tool_call_id'), None)
        if pair is None:
            gaps['unmatched_tool_messages'] += 1
            continue  # Preserved partial histories may lack the matching call.
        call_index, function, origin = pair
        content = message.get('content', '')
        try:
            response = json.loads(content) if isinstance(content, str) else content
        except ValueError:
            response = None  # Bounded-context omission marker, not zero reads.
        if not isinstance(response, dict) or not response:
            gaps['unavailable_or_unparseable_responses'] += 1
            response = {}
        later = next((i for i in model_indexes if i > index), None)
        author_relation = relation(response.get('author'), agent, settings)
        body = isinstance(response.get('text'), str) and bool(response['text'].strip())
        if body and author_relation == 'unknown':
            gaps['unclassified_body_responses_including_sources_and_directories'] += 1
        successful_peer_body = body and author_relation == 'peer' and 'error' not in response
        observed.append({'reference': {'file': source, 'agent': agent, 'message_index': index,
                                      'call_index': call_index},
            'question_round': number, 'agent': agent, 'origin': origin,
            'operation': function.get('name'), 'url': response.get('url'),
            'author': response.get('author'), 'author_relation': author_relation,
            'response_available': bool(response), 'success': 'error' not in response if response else None,
            'body_present': body if response else None,
            'peer_body': successful_peer_body,
            'subsequent_model_message_index': later,
            'receipt': successful_peer_body and later is not None})
    gaps['unmatched_assistant_tool_calls'] += len(pending)
    return observed, dict(gaps)


def evaluate_run(root):
    artifacts = Artifacts(root)
    settings = artifacts.read('settings.json', required=True)
    if not isinstance(settings, dict) or settings.get('schema') != 'synchronized-exchange-v1':
        raise ValueError(f'{root}: settings must use synchronized-exchange-v1')
    for field in ('visible_labels', 'notebooks', 'provenance'):
        if field in settings and not isinstance(settings[field], dict):
            raise ValueError(f'{root}: settings.{field} must be an object')
    if any(not isinstance(v, str) for v in settings.get('notebooks', {}).values()):
        raise ValueError(f'{root}: notebook slugs must be strings')
    manifest = artifacts.read('manifest.json')
    if manifest is not None and not isinstance(manifest, dict):
        raise ValueError(f'{root}: manifest must be an object')
    results = artifacts.read('results.json')
    if results is not None:
        objects(results, 'results.json')
    events = artifacts.read('evidence-index.host-only.jsonl', lines=True)
    if events is not None:
        objects(events, 'evidence index')
    if results is None and events is None:
        raise ValueError(f'{root}: requires results.json or evidence-index.host-only.jsonl')
    warnings = []
    unique, seen = [], {}
    for position, event in enumerate(events or []):
        line = artifacts.line_numbers['evidence-index.host-only.jsonl'][position]
        key = event.get('event_id')
        if key is not None and not isinstance(key, str):
            raise ValueError('event_id must be a string')
        if key in seen:
            if seen[key] != event:
                raise ValueError(f'Conflicting duplicate event ID: {key}')
            warnings.append(f'Duplicate event ID ignored: {key}')
            continue
        if key is not None:
            seen[key] = event
        unique.append({**event, 'reference': {'file': 'evidence-index.host-only.jsonl', 'line': line,
                                             'event_id': key}})
    events = unique
    phases = []
    runtime_requests = []
    for index, row in enumerate(results or []):
        requests = row.get('model_requests')
        if requests is not None:
            objects(requests, f'results.json/{index}/model_requests')
            for request_index, request in enumerate(requests):
                identity = {k: v for k, v in request.items() if k in
                            ('model', 'model_name', 'model_digest', 'digest', 'revision', 'backend', 'backend_version')}
                if identity:
                    runtime_requests.append({'reference': {'file': 'results.json', 'index': index,
                                                           'request_index': request_index}, 'identity': identity})
        phase = {k: row.get(k) for k in ('agent', 'question_id', 'phase_role', 'status', 'answer',
                 'generated_tokens_observed', 'token_accounting_complete', 'browser_calls', 'host_browser_calls',
                 'limits_reached', 'safety_timeout_hit', 'note_preservation')}
        phase.update(reference={'file': 'results.json', 'index': index},
                     model_request_count=len(requests) if requests is not None else None,
                     transcript_reference=row.get('log_path'))
        phase_events = [e for e in events if e.get('event') == 'phase_finished' and e.get('results_index') == index]
        phase['question_round'] = phase_events[0].get('question_round') if len(phase_events) == 1 else None
        phase['stage'] = phase_events[0].get('stage') if len(phase_events) == 1 else None
        phase['event_coverage'] = len(phase_events)
        phases.append(phase)
    groups = defaultdict(list)
    for phase in phases:
        groups[(str(phase['agent']), str(phase['phase_role']))].append(phase)
    costs = [{'agent': agent, 'phase_role': role, 'phases': len(rows),
              'generated_tokens': tally(rows, 'generated_tokens_observed'),
              'native_accounting_complete_rows': sum(r['token_accounting_complete'] is True for r in rows),
              'model_browser_calls': tally(rows, 'browser_calls'),
              'host_browser_calls': tally(rows, 'host_browser_calls'),
              'model_requests': tally(rows, 'model_request_count')}
             for (agent, role), rows in sorted(groups.items())]
    append_events = []
    for event in events:
        if event.get('event') != 'append_attempt':
            continue
        append_events.append({**event, 'destination_relation': destination_relation(
            event.get('destination'), event.get('agent'), settings),
            'author_relation': relation(event.get('author'), event.get('agent'), settings)})
    append_counts = Counter((str(e.get('actor', 'unknown')),
                            'success' if e.get('success') is True else 'failure' if e.get('success') is False else 'unknown',
                            e['destination_relation']) for e in append_events)
    observations, histories, history_gaps = [], [], {}
    for path in sorted(root.glob('round-*-histories-before-reset.json')):
        match = re.fullmatch(r'round-(\d+)-histories-before-reset.json', path.name)
        if not match:
            continue
        history = artifacts.read(path.name)
        if not isinstance(history, dict):
            raise ValueError(f'{path}: history must be an object')
        histories.append(path.name)
        for agent, messages in history.items():
            found, gaps = history_observations(messages, agent, int(match[1]), path.name, settings)
            observations.extend(found)
            history_gaps[f'{path.name}/{agent}'] = gaps
    # Final histories can be post-reset and resumed prefixes may live elsewhere.
    # Do not count them again or pretend they cover all prior questions.
    source_observations = [o for o in observations if o['origin'] == 'model' and
                           o['operation'] in ('open', 'click') and
                           isinstance(o['url'], str) and o['url'].startswith('https://docs.test/')
                           and '/request-history' not in o['url']]
    peer_bodies = [o for o in observations if o['peer_body']]
    counts = {name: sum(predicate(o) for o in peer_bodies) if histories else None for name, predicate in {
        'host_peer_body_deliveries': lambda o: o['origin'] == 'host',
        'host_peer_body_receipts_with_later_model_message': lambda o: o['origin'] == 'host' and o['receipt'],
        'voluntary_peer_body_reads': lambda o: o['origin'] == 'model' and o['success'] is True,
        'voluntary_peer_body_receipts_with_later_model_message': lambda o: o['origin'] == 'model' and o['receipt']}.items()}
    expected = settings.get('maximum_phases')
    if manifest and nonnegative(manifest.get('next_phase_index')) and results is not None and manifest['next_phase_index'] != len(results):
        warnings.append('Manifest next_phase_index differs from results length; interrupted or inconsistent snapshot.')
    if manifest and manifest.get('status') == 'complete' and nonnegative(expected) and len(phases) != expected:
        warnings.append('Complete manifest conflicts with configured phase count.')
    uncovered = [i for i, phase in enumerate(phases) if phase['event_coverage'] != 1]
    if uncovered:
        warnings.append('Some result phases lack unique local event coverage; may include copied resume prefix or incomplete index.')
    migration = settings.get('note_policy_migration')
    provenance = settings.get('provenance', {})
    report = {
        'run_directory': str(root),
        'provenance': {'configured': {k: settings.get(k) for k in
            ('model', 'model_digest', 'inference_profile', 'intended_model', 'source_access',
             'source_access_policy', 'no_peer_information', 'note_retry_policy')},
            'saved_settings_reference': {'file': 'settings.json'},
            'source_hashes': settings.get('source_hashes'),
            'seed': settings.get('seed'),
            'saved_policy': settings.get('policy'),
            'saved_configuration': {k: v for k, v in settings.items() if k not in ('system_prompts', 'provenance')},
            'recorded_runtime_inspection': provenance.get('model'),
            'recorded_provenance': provenance, 'request_identity_evidence': runtime_requests,
            'policy': 'Recorded identities only; inspection metadata is a saved claim, not independent weight verification. No current defaults or cross-run pooling.'},
        'coverage': {'manifest': manifest, 'results_present': results is not None,
            'event_index_present': 'evidence-index.host-only.jsonl' in artifacts.inventory,
            'configured_phase_count': expected, 'observed_result_phases': len(phases) if results is not None else None,
            'phase_status_counts': dict(Counter(str(p['status']) for p in phases)),
            'submitted_nonblank_answers': sum(p['phase_role'] == 'answer' and isinstance(p['answer'], str) and bool(p['answer'].strip()) for p in phases) if results is not None else None,
            'expected_answer_slots': sum(len(ids) for ids in settings['question_ids'].values()) if isinstance(settings.get('question_ids'), dict) and all(isinstance(ids, list) for ids in settings['question_ids'].values()) else None,
            'agent_question_pairs_in_results': len({(str(p['agent']), str(p['question_id'])) for p in phases}) if phases else None,
            'history_files_reviewed': histories, 'result_indices_without_unique_local_event': uncovered,
            'lineage': {'recorded_resume': settings.get('resume'), 'note_policy_migration': migration,
                       'parent_prefix_verified': False,
                       'scope': 'Results are this saved trajectory; local events/histories may cover only its resumed suffix. Abandoned parent attempts are not included. No cross-run totals.'},
            'warnings': warnings},
        'phases': phases, 'costs_by_agent_phase': costs,
        'append_attempts': {'scope': 'Local indexed persistence attempts, not note-generation attempts; truncated generations may never call append.',
            'observed_total': len(append_events) if 'evidence-index.host-only.jsonl' in artifacts.inventory else None,
            'counts': [{'actor': actor, 'outcome': outcome, 'destination_relation': dest, 'count': n}
                       for (actor, outcome, dest), n in sorted(append_counts.items())], 'events': append_events},
        'peer_information': {'scope': 'Only reviewed per-question histories; body receipt requires a later non-host assistant message in that same history.',
            'history_coverage_gaps': history_gaps,
            'count_coverage': 'Observed-only lower bounds in retained round histories; gaps and omitted histories are not negative evidence.',
            'opportunity_denominator': None,
            'opportunity_status': 'not_applicable' if settings.get('no_peer_information') is True else 'unknown',
            'counts': counts, 'history_observations': observations,
            'host_exposure_events': [e for e in events if e.get('event') in ('host_notebook_exposure', 'host_metadata_log_exposure')],
            'indexed_tool_observations': [e for e in events if e.get('event') == 'tool_observation']},
        'source_access': {'observed_successful_source_returns': sum(o['success'] is True for o in source_observations) if histories else None,
            'observations': source_observations,
            'limitations': 'Denied responses often omit URL, so source-denial counts and access eligibility are unknown. Returned text does not establish answer support.'},
        'memory_and_barriers': [e for e in events if e.get('event') in
            ('self_memory_restored', 'question_context_reset', 'stage_snapshots_frozen', 'stage_published')],
        'failures': [e for e in events if e.get('event') == 'run_failed'],
        'annotation_template': [{'evidence_references': [o['reference']], 'entry_url': o.get('url', o.get('entry_url')),
            'states': {field: 'unassessed' for field in SEMANTIC_FIELDS}, 'annotator': None,
            'claim_and_exact_quote': None, 'additional_evidence_references': [], 'reply_to_evidence_references': [], 'limitations': None}
            for o in peer_bodies + append_events + [{'reference': {'file': 'results.json' if results is not None else 'evidence-index.host-only.jsonl', 'scope': 'run; replace with exact span before assessing'}}]],
        'artifact_sha256': artifacts.inventory, 'missing_artifacts': artifacts.missing,
    }
    return report


def markdown(report):
    lines = ['# Saved 9d mechanical evaluation', '',
             'Each input is separate. Observed counts describe retained evidence, not semantic collaboration or causal benefit.', '']
    for run in report['runs']:
        coverage = run['coverage']
        lines.extend([f"## {run['run_directory']}", '',
            f"Recorded configuration: `{json.dumps(run['provenance']['configured'], ensure_ascii=False)}`", '',
            f"Recorded runtime inspection: `{json.dumps(run['provenance']['recorded_runtime_inspection'], ensure_ascii=False)}`", '',
            f"Manifest: `{json.dumps(coverage['manifest'])}`", '',
            f"Observed result phases: {coverage['observed_result_phases']}; configured: {coverage['configured_phase_count']}", '',
            f"Submitted nonblank answers: {coverage['submitted_nonblank_answers']}; expected slots: {coverage['expected_answer_slots']} (not a correctness score)", '',
            '| Agent | Phase | Observed tokens | Known / all rows |', '| --- | --- | ---: | ---: |'])
        for cost in run['costs_by_agent_phase']:
            token = cost['generated_tokens']
            lines.append(f"| {cost['agent']} | {cost['phase_role']} | {token['observed_sum']} | {token['known_rows']} / {token['total_rows']} |")
        lines.extend(['', f"Indexed append persistence attempts: {run['append_attempts']['observed_total']}", '',
                      f"Append outcomes by actor and destination: `{json.dumps(run['append_attempts']['counts'])}`", '',
                      f"Reviewed history files: {len(coverage['history_files_reviewed'])}; observed-only lower bounds.", '',
                      f"History coverage gaps: `{json.dumps(run['peer_information']['history_coverage_gaps'])}`", '',
                      f"Peer-body evidence: `{json.dumps(run['peer_information']['counts'])}`", '',
                      f"Opportunity denominator: unknown; status: {run['peer_information']['opportunity_status']}", '',
                      'Parent prefix is unverified. Results may include copied phases; local histories/index may cover a suffix. Never add these run totals together.', ''])
        lines.extend(f'- {warning}' for warning in coverage['warnings'])
        lines.append('')
    lines.extend(['Semantic annotation states: supported, contradicted, insufficient-evidence, unassessed. Templates in JSON are unassessed; ingestion/adjudication is not implemented.', '',
                  'Capture timestamps are index-write times, not action times. Entry created_at is preserved separately. No elapsed response latency is inferred.', ''])
    return '\n'.join(lines)


def evaluate(run_dirs, output):
    roots = [Path(path).resolve() for path in run_dirs]
    if not roots or any(not root.is_dir() for root in roots):
        raise ValueError('Every input must be an existing saved-run directory')
    if len(set(roots)) != len(roots):
        raise ValueError('Duplicate input run directory')
    if any(a != b and a.is_relative_to(b) for a in roots for b in roots):
        raise ValueError('Nested run inputs are ambiguous')
    destination = Path(output).absolute()
    if destination.exists() or destination.is_symlink():
        raise ValueError('Output must be a fresh directory')
    destination = destination.resolve()
    if not destination.parent.is_dir():
        raise ValueError('Output parent must already exist')
    if any(destination.is_relative_to(root) or root.is_relative_to(destination) for root in roots):
        raise ValueError('Output must be outside input run directories')
    runs = [evaluate_run(root) for root in roots]
    report = {'schema': SCHEMA, 'aggregation': 'per-input only; overlapping trajectories must not be summed', 'runs': runs}
    encoded = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    readable = markdown(report)
    # All input parsing and serialization precedes output creation. Keep partial
    # output on an actual write failure, so the failure remains diagnosable.
    destination.mkdir()
    with (destination / 'evaluation.host-only.json').open('x') as stream:
        stream.write(encoded)
    with (destination / 'evaluation.host-only.md').open('x') as stream:
        stream.write(readable)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, epilog='No models, network, SQLite writes, or cross-run pooling. Missing evidence stays unknown.')
    parser.add_argument('run_dirs', nargs='+', type=Path, help='Saved synchronized 9d directories (settings + results and/or event index)')
    parser.add_argument('--output', required=True, type=Path, help='Fresh host-only output directory outside all inputs; parent must exist')
    args = parser.parse_args(argv)
    try:
        evaluate(args.run_dirs, args.output)
    except (ValueError, OSError, TypeError) as error:
        parser.exit(2, f'error: {error}\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
