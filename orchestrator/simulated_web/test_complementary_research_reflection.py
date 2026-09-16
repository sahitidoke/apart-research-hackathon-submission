"""Host-only 10c entrypoint parity; no model or cloud requests."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from orchestrator.simulated_web import modal_hf_complementary_research_reflection as cli
from orchestrator.simulated_web.concurrent_research_reflection import build_settings
from orchestrator.simulated_web.browser import Browser
from orchestrator.simulated_web.source_access import access_browser_options
from orchestrator.simulated_web.source_discovery import browser_discovery

BASE=Path(__file__).resolve().parents[2]/'research-log/mlb-tokenpair-2026-09-12'


def inputs():
    def read(path):return json.loads((BASE/path).read_text())
    return dict(records=[json.loads(line) for line in (BASE/'variant-009d/dataset.host-only.jsonl').read_text().splitlines() if line.strip()],
        topic=(BASE/'variant-011a/topic-research.txt').read_text().strip(),
        selectors=read('variant-009d/editable-sources.host-only.json'),
        question_ids=read('variant-011a/question-ids-same-2.host-only.json'),
        access_manifest=read('variant-011b/access-manifest.host-only.json'),
        visible_labels=read('variant-009d/visible-labels.host-only.json'),
        round_leaders=read('variant-011a/round-leaders-2.host-only.json'),
        answer_final_reserve_tokens=512,inference_profile='hf-fp8-v1',note_retry_policy=cli.SHORT_NOTE_POLICY,
        source_access_policy=cli.DISCOVERY_ACCESS_POLICY,notebook_context_policy='9e-v1',
        notebook_quota_policy='notebook-exempt-v1',question_pairing_policy='same-question-v1')


def test_actual_inputs_exact_reference_runner_and_resources():
    data=inputs();settings=cli.validate_fresh('test-fp8-concurrent-10c',data)
    reference,pages,_=build_settings(**data)
    assert settings==reference
    resource=cli.resources(settings)
    assert resource['maximum_combined_generated_tokens']==54272
    assert resource['initial_research_tokens']==8192 and resource['answer_tokens']==2048 and resource['reflection_tokens']==4096
    assert resource['answer_seconds']==180 and resource['coordinator_readiness']['initial_cold_seconds']==600
    assert resource['phase_records']==16 and resource['normal_model_phases']==12
    assert settings['question_ids']['agent-1']==settings['question_ids']['agent-2']==data['access_manifest']['partition_audit']['selected_questions']
    browser=Browser(pages,':memory:',**browser_discovery(settings['discovery_plan']),**access_browser_options(settings['access_plan']))
    try:
        groups=settings['discovery_plan']['groups']
        for agent in settings['question_ids']:
            foreign={url for group,urls in groups.items() if group not in data['access_manifest']['source_groups'][agent] for url in urls}
            assert foreign
            for url in settings['discovery_plan']['listing_urls']:
                assert not foreign & {link['url'] for link in browser.call(agent,'open',{'url':url}).get('links',[])}
            assert not foreign & {row['url'] for row in browser.search('baseball',agent)['results']}
            assert 'error' not in browser.call(agent,'open',{'url':sorted(foreign)[0]})
        visible=json.dumps(pages)
        assert 'partition_audit' not in visible and 'is_supporting' not in visible and 'terminal_owner' not in visible
    finally:browser.close()


def test_entrypoint_rejects_identity_partition_question_and_count_mismatch():
    data=inputs()
    with pytest.raises(ValueError,match='end in'):cli.validate_fresh('test-fp8-concurrent-10b',data)
    with pytest.raises(ValueError,match='Question count'):cli.validate_fresh('test-fp8-concurrent-10c',data,6)
    changed=deepcopy(data);changed['access_manifest']['schema']='source-access-v1'
    with pytest.raises(ValueError,match='audited complementary'):cli.validate_fresh('test-fp8-concurrent-10c',changed)
    changed=deepcopy(data);changed['question_ids']={a:list(reversed(ids)) for a,ids in changed['question_ids'].items()}
    with pytest.raises(ValueError,match='Selected questions'):cli.validate_fresh('test-fp8-concurrent-10c',changed)
    changed=deepcopy(data);changed['access_manifest']['dataset_sha256']='bad'
    with pytest.raises(ValueError,match='Invalid'):cli.validate_fresh('test-fp8-concurrent-10c',changed)
