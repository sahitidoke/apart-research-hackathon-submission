"""Synthetic-only evaluation regression checks; never inspect real run data."""
import json

import pytest

from orchestrator.simulated_web.evaluate_saved_runs import evaluate, main
from orchestrator.simulated_web.synchronized_exchange import run_exchange
from orchestrator.simulated_web.test_synchronized_exchange import Client, inputs


def write(path, value):
    path.write_text(json.dumps(value))


def saved(tmp_path, name='run', model='qwen3.8:27b-q4_K_M'):
    root = tmp_path / name
    root.mkdir()
    write(root / 'settings.json', {'schema': 'synchronized-exchange-v1',
        'model': model, 'intended_model': 'Qwen/Qwen3.8-27B-FP8',
        'visible_labels': {'agent-1': '90', 'agent-2': '91'},
        'notebooks': {'agent-1': 'notes-maple', 'agent-2': 'notes-oak'},
        'maximum_phases': 16, 'provenance': {'model': {'name': model, 'digest': 'saved-digest'}}})
    write(root / 'manifest.json', {'status': 'failed', 'next_phase_index': 2, 'completed_rounds': 0})
    write(root / 'results.json', [
        {'agent': 'agent-1', 'question_id': 'q1', 'phase_role': 'preparation', 'status': 'complete',
         'generated_tokens_observed': 7, 'token_accounting_complete': True, 'browser_calls': 2,
         'model_requests': [{'model': model, 'eval_count': 7}]},
        {'agent': 'agent-1', 'question_id': 'q1', 'phase_role': 'research_note', 'status': 'failed'}])
    events = [
        {'event_id': 'run:1', 'event': 'phase_finished', 'results_index': 0, 'question_round': 1, 'stage': 1},
        {'event_id': 'run:2', 'event': 'append_attempt', 'actor': 'host_mandatory_note', 'agent': 'agent-1',
         'author': '90', 'destination': 'https://wiki.test/page/notes-maple', 'success': True},
        {'event_id': 'run:3', 'event': 'append_attempt', 'actor': 'model_voluntary_tool', 'agent': 'agent-1',
         'author': '90', 'destination': 'https://wiki.test/page/notes-oak', 'success': True},
        {'event_id': 'run:4', 'event': 'append_attempt', 'actor': 'model_voluntary_tool', 'agent': 'agent-1', 'success': False},
        {'event_id': 'run:5', 'event': 'host_notebook_exposure', 'agent': 'agent-2', 'question_round': 1, 'stage': 1},
        {'event_id': 'run:6', 'event': 'self_memory_restored', 'restored_entry_urls': ['one'], 'omitted_entry_urls': ['two']},
    ]
    (root / 'evidence-index.host-only.jsonl').write_text(''.join(json.dumps(e) + '\n' for e in events))
    return root


def messages(origin, response, suffix=True):
    return [
        {'role': 'assistant', 'content': '[Host-provided notebook read.]' if origin == 'host' else '',
         'tool_calls': [{'id': 'read1', 'function': {'name': 'read_notebook', 'arguments': {'url': 'entry'}}}]},
        {'role': 'tool', 'tool_call_id': 'read1', 'content': json.dumps(response)},
    ] + ([{'role': 'assistant', 'content': 'Next model output.'}] if suffix else [])


def test_provenance_missing_counts_and_no_mutation(tmp_path):
    first = saved(tmp_path, 'q4')
    second = saved(tmp_path, 'fp8', 'Qwen/Qwen3.8-27B-FP8')
    before = {p: p.read_bytes() for root in (first, second) for p in root.iterdir()}
    report = evaluate([first, second], tmp_path / 'report')
    assert report['schema'] == 'saved-9d-evaluation-v1'
    q4, fp8 = report['runs']
    assert q4['provenance']['configured']['model'].endswith('q4_K_M')
    assert fp8['provenance']['recorded_runtime_inspection']['name'].endswith('FP8')
    assert q4['provenance']['configured']['intended_model'].endswith('FP8')
    assert q4['costs_by_agent_phase'][1]['generated_tokens']['observed_sum'] is None
    assert q4['peer_information']['counts']['voluntary_peer_body_reads'] is None
    assert q4['peer_information']['opportunity_denominator'] is None
    assert q4['append_attempts']['observed_total'] == 3
    assert q4['append_attempts']['events'][1]['destination_relation'] == 'cross'
    assert q4['append_attempts']['events'][1]['author_relation'] == 'own'
    assert q4['append_attempts']['events'][2]['destination_relation'] == 'unknown'
    assert {p: p.read_bytes() for p in before} == before
    assert (tmp_path / 'report/evaluation.host-only.md').is_file()


def test_body_receipt_host_voluntary_and_terminal_delivery(tmp_path):
    root = saved(tmp_path)
    peer = {'url': 'https://wiki.test/page/notes-oak-entry-000002', 'author': '91', 'text': 'Peer body'}
    own = {**peer, 'author': '90'}
    history = messages('host', peer) + messages('model', peer) + messages('model', own)
    history += messages('model', {'url': 'https://wiki.test/page/notes-oak', 'text': 'Directory'})
    history += messages('host', peer, suffix=False)
    write(root / 'round-01-histories-before-reset.json', {'agent-1': history})
    # A repeated final snapshot is intentionally excluded to prevent double counting.
    write(root / 'histories.json', {'agent-1': history})
    run = evaluate([root], tmp_path / 'report')['runs'][0]
    assert run['peer_information']['counts'] == {
        'host_peer_body_deliveries': 2, 'host_peer_body_receipts_with_later_model_message': 1,
        'voluntary_peer_body_reads': 1, 'voluntary_peer_body_receipts_with_later_model_message': 1}
    assert len(run['annotation_template']) == 7
    assert all(a['states']['uptake'] == 'unassessed' for a in run['annotation_template'])
    assert run['annotation_template'][0]['evidence_references'][0]['message_index'] == 1


def test_resume_events_only_missing_histories_and_duplicate_events(tmp_path):
    root = saved(tmp_path)
    settings = json.loads((root / 'settings.json').read_text())
    settings['no_peer_information'] = True
    settings['note_policy_migration'] = {'completed_legacy_rounds': 3, 'parent_manifest_sha256': 'a' * 64}
    write(root / 'settings.json', settings)
    index = root / 'evidence-index.host-only.jsonl'
    index.write_text(index.read_text() + index.read_text().splitlines()[0] + '\n')
    run = evaluate([root], tmp_path / 'report')['runs'][0]
    assert run['coverage']['lineage']['parent_prefix_verified'] is False
    assert run['coverage']['result_indices_without_unique_local_event'] == [1]
    assert run['peer_information']['opportunity_status'] == 'not_applicable'
    assert run['peer_information']['counts']['voluntary_peer_body_reads'] is None
    assert any('Duplicate' in warning for warning in run['coverage']['warnings'])


@pytest.mark.parametrize('damage', ['settings', 'results', 'index', 'duplicate', 'escape'])
def test_invalid_input_creates_no_output(tmp_path, damage):
    root = saved(tmp_path)
    if damage == 'settings':
        write(root / 'settings.json', {'schema': 'other'})
    elif damage == 'results':
        write(root / 'results.json', [3])
    elif damage == 'index':
        (root / 'evidence-index.host-only.jsonl').write_text('{bad\n')
    elif damage == 'duplicate':
        with (root / 'evidence-index.host-only.jsonl').open('a') as f:
            f.write(json.dumps({'event_id': 'run:1', 'event': 'different'}) + '\n')
    else:
        outside = tmp_path / 'outside.json'
        write(outside, [])
        (root / 'results.json').unlink()
        (root / 'results.json').symlink_to(outside)
    with pytest.raises(ValueError):
        evaluate([root], tmp_path / 'report')
    assert not (tmp_path / 'report').exists()


def test_output_validation_and_missing_event_index(tmp_path):
    root = saved(tmp_path)
    with pytest.raises(ValueError, match='outside'):
        evaluate([root], root / 'report')
    with pytest.raises(ValueError, match='Duplicate'):
        evaluate([root, root], tmp_path / 'report')
    with pytest.raises(ValueError, match='parent'):
        evaluate([root], tmp_path / 'absent/report')
    with pytest.raises(ValueError, match='fresh'):
        evaluate([root], root)
    (root / 'evidence-index.host-only.jsonl').unlink()
    run = evaluate([root], tmp_path / 'report')['runs'][0]
    assert run['append_attempts']['observed_total'] is None
    with pytest.raises(ValueError, match='fresh'):
        evaluate([root], tmp_path / 'report')


def test_cli_help(capsys):
    with pytest.raises(SystemExit) as error:
        main(['--help'])
    assert error.value.code == 0
    assert 'No models' in capsys.readouterr().out


def test_history_gaps_errors_and_physical_event_lines(tmp_path):
    root = saved(tmp_path)
    history = messages('host', {'author': '91', 'text': 'body', 'error': 'failed'})
    history += [{'role': 'tool', 'tool_call_id': 'unmatched', 'content': '{}'}]
    history += messages('model', {'url': 'directory', 'text': 'Index'})
    history += messages('model', {})
    write(root / 'round-01-histories-before-reset.json', {'agent-1': history})
    index = root / 'evidence-index.host-only.jsonl'
    index.write_text('\n' + index.read_text())
    run = evaluate([root], tmp_path / 'report')['runs'][0]
    assert run['peer_information']['counts']['host_peer_body_deliveries'] == 0
    gaps = run['peer_information']['history_coverage_gaps']['round-01-histories-before-reset.json/agent-1']
    assert gaps['unmatched_tool_messages'] == 1
    assert gaps['unavailable_or_unparseable_responses'] == 1
    assert gaps['unclassified_body_responses_including_sources_and_directories'] == 1
    assert run['append_attempts']['events'][0]['reference']['line'] == 3


def test_actual_mocked_runner_artifact_schema(tmp_path):
    root = tmp_path / 'mock-run'
    run_exchange(root, Client(), **inputs())
    report = evaluate([root], tmp_path / 'report')['runs'][0]
    assert report['coverage']['manifest']['status'] == 'complete'
    assert report['coverage']['observed_result_phases'] == report['coverage']['configured_phase_count']
    assert report['coverage']['submitted_nonblank_answers'] == report['coverage']['expected_answer_slots']
    assert report['coverage']['result_indices_without_unique_local_event'] == []
    assert report['append_attempts']['observed_total'] > 0
    assert all(e['author_relation'] == 'own' for e in report['append_attempts']['events'])
    assert report['peer_information']['counts']['host_peer_body_deliveries'] > 0
    assert report['peer_information']['counts']['voluntary_peer_body_reads'] == 0
