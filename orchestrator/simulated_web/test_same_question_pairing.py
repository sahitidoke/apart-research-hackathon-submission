"""Synthetic same-question opt-in tests; preserve the synchronized 10a protocol."""
import hashlib
import json

import pytest

from orchestrator.simulated_web.synchronized_exchange import build_settings, run_exchange, load_checkpoint, schedule
from orchestrator.simulated_web.synchronized_notebooks import make_browser
from orchestrator.simulated_web.test_9e import options, NineClient


def paired_options():
    values=options()
    # Choose the existing first actor's question in each synthetic round.
    values['question_ids']={agent:list(values['question_ids']['agent-1']) for agent in ('agent-1','agent-2')}
    return {**values,'notebook_quota_policy':'notebook-exempt-v1','question_pairing_policy':'same-question-v1'}


def test_same_question_preserves_stage_tools_access_and_budgets():
    legacy,pages,_=build_settings(**options(),notebook_quota_policy='notebook-exempt-v1')
    same,same_pages,_=build_settings(**paired_options())
    assert same['question_pairing_policy']=='same-question-v1'
    assert same['question_ids']['agent-1']==same['question_ids']['agent-2']
    for key in ('policy','synchronized_policy','system_prompts','notebook_tool_schemas','source_access','access_plan','discovery_plan',
                'maximum_phases','maximum_generated_tokens','maximum_combined_generated_tokens','answer_support_verifier',
                'notebook_quota_policy','related_append_only'):
        assert same[key]==legacy[key],key
    assert same_pages==pages
    rows=schedule(same)
    assert len(rows)==32 and {row[1] for row in rows}=={1,2,3,4}
    for round_index in (1,2):
        assert len({row[4] for row in rows if row[0]==round_index})==1
    browser=make_browser(same,pages,':memory:')
    try:
        root='https://wiki.test/page/'+same['notebooks']['agent-2']
        saved=browser.call('agent-1','append_notebook',{'text':'Cross-note','notebook':root})
        assert saved['author']==same['visible_labels']['agent-1']
        assert browser.call('agent-2','read_notebook',{'url':saved['saved'],'revision':''})['text']=='Cross-note'
    finally:browser.close()


def test_legacy_different_rule_and_optin_same_rule_are_strict(tmp_path):
    values=paired_options();values.pop('question_pairing_policy')
    with pytest.raises(ValueError,match='different question IDs'):build_settings(**values)
    for value in ('same-question-v1','bad',True):
        values={**options(),'question_pairing_policy':value}
        with pytest.raises(ValueError):run_exchange(tmp_path/str(value),NineClient({}),**values)
        assert not (tmp_path/str(value)).exists()


def test_saved_pairing_resume_and_tamper_validation(tmp_path):
    def stop(checkpoint):
        if checkpoint.name=='rounds-001':raise RuntimeError('synthetic stop')
    with pytest.raises(RuntimeError,match='synthetic stop'):
        run_exchange(tmp_path/'first',NineClient({}),**paired_options(),checkpoint_callback=stop)
    checkpoint=tmp_path/'first/checkpoints/rounds-001'
    data,browser,_=load_checkpoint(checkpoint);browser.close()
    assert data['settings.json']['question_pairing_policy']=='same-question-v1'
    assert run_exchange(tmp_path/'resumed',NineClient({}),resume_from=checkpoint)['status']=='complete'
    final,browser,_=load_checkpoint(tmp_path/'resumed/checkpoints/rounds-002');browser.close()
    assert final['settings.json']['question_pairing_policy']=='same-question-v1'
    with pytest.raises(ValueError,match='Resume overrides'):
        run_exchange(tmp_path/'override',NineClient({}),resume_from=checkpoint,question_pairing_policy='same-question-v1')
    # Update checksum too: semantic reconstruction must reject an unsupported policy.
    path=checkpoint/'settings.json';settings=json.loads(path.read_text());settings['question_pairing_policy']='bad'
    path.write_text(json.dumps(settings))
    path=checkpoint/'synchronized-checkpoint.json';manifest=json.loads(path.read_text())
    manifest['files_sha256']['settings.json']=hashlib.sha256((checkpoint/'settings.json').read_bytes()).hexdigest()
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='pairing policy'):load_checkpoint(checkpoint)
