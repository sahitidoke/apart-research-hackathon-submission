"""11c log-removal boundaries on populated notebooks and collaboration phases."""
import json
from unittest.mock import patch

import pytest

from orchestrator.simulated_web import modal_hf_no_log_collaboration_pilot as cli
from orchestrator.simulated_web import concurrent_collaboration_pilot as runner
from orchestrator.simulated_web.synchronized_notebooks import make_browser
from orchestrator.simulated_web.test_complementary_research_reflection import inputs as original_inputs
from orchestrator.simulated_web.test_collaboration_pilot_11b2 import Final512Owner
from orchestrator.simulated_web.test_same_question_pairing import paired_options


def inputs():
    return {**original_inputs(),'coordinator_readiness_policy':'cold-start-600-v2'}


def test_actual_input_settings_parity_and_required_optin():
    original=inputs();settings,pages,_=runner.build_settings(**original)
    changed={**original,'agent_log_policy':'no-agent-history-v1'}
    after=cli.validate_fresh('pilot-fp8-concurrent-11c',changed)
    assert {k for k in set(settings)|set(after) if settings.get(k)!=after.get(k)}=={'system_prompts','policy','agent_log_policy','log_exposure','request_history','history_search','history_search_policy'}
    assert {k for k in settings['policy'] if settings['policy'][k]!=after['policy'][k]}=={'request_history_mode'}
    assert after['policy']['request_history_mode']=='disabled'
    obsolete='Superseded automatic request-history metadata is archived; the newest payload remains. '
    for agent,prompt in settings['system_prompts'].items():
        assert obsolete in prompt
        assert after['system_prompts'][agent]==prompt.replace(obsolete,'')
    assert after['maximum_combined_generated_tokens']==54272
    assert cli.resources(after)['answer_final_reserve_tokens']==512
    with pytest.raises(ValueError,match='no agent-visible'):cli.validate_fresh('pilot-fp8-concurrent-11c',original)
    with pytest.raises(ValueError,match='Invalid agent log'):runner.build_settings(**original,agent_log_policy='unknown')


def test_populated_history_hidden_notebooks_search_read_crossappend_paginate_and_audit(tmp_path):
    settings,pages,_=runner.build_settings(**inputs(),agent_log_policy='no-agent-history-v1')
    browser=make_browser(settings,pages,tmp_path/'wiki.sqlite3')
    try:
        # Populate both histories and more than one notebook-directory page.
        roots=settings['notebooks'];saved=[]
        for i in range(7):
            result=browser.call('agent-2','append_notebook',{'text':f'uniquepeerword entry{i}','notebook':'https://wiki.test/page/'+roots['agent-2']})
            assert 'error' not in result;saved.append(result['saved'])
        before=browser.db.execute('SELECT count(*) FROM audit').fetchone()[0]
        assert browser.db.execute('SELECT count(*) FROM request_events').fetchone()[0]>=7
        for agent in roots:
            root=browser.call(agent,'open',{'url':'https://docs.test/'})
            assert not any('request-history' in link['url'] for link in root.get('links',[]))
            # Known URL, fake legacy window, and a preexisting click reference cannot reopen history.
            for url in ('https://docs.test/request-history','https://docs.test/request-history?window=old&offset=0&position=0','https://docs.test/request-history?owner=agent-1','https://docs.test/request-history?agent=agent-2','https://docs.test/request-history?query=uniquepeerword'):
                assert 'error' in browser.call(agent,'open',{'url':url})
            browser.views.setdefault(agent,{})['old-link']=[{'label':'History','url':'https://docs.test/request-history'}]
            assert 'error' in browser.call(agent,'click',{'page_id':'old-link','link_id':1})
            assert browser.history_search_candidates(agent)==[]
            assert 'error' in browser.call(agent,'history',{})
            groups=settings['discovery_plan']['groups']
            foreign={url for group,urls in groups.items() if group not in settings['access_manifest']['source_groups'][agent] for url in urls}
            assert 'error' not in browser.call(agent,'open',{'url':sorted(foreign)[0]})
            results=browser.call(agent,'search',{'query':'uniquepeerword'})['results']
            assert any(row['url'] in saved for row in results)
            assert not any('request-history' in row['url'] for row in results)
        directory=browser.call('agent-1','open',{'url':'https://wiki.test/page/'+roots['agent-2']})
        assert any('?offset=5' in link['url'] for link in directory['links'])
        page=browser.call('agent-1','open',{'url':'https://wiki.test/page/'+roots['agent-2']+'?offset=5'})
        assert any(link['url']==saved[-1] for link in page['links'])
        own=browser.call('agent-2','read_notebook',{'url':saved[0],'revision':''})
        assert 'uniquepeerword' in json.dumps(own)
        read=browser.call('agent-1','read_notebook',{'url':saved[0],'revision':''})
        assert 'error' not in read
        assert 'uniquepeerword' in json.dumps(read)
        cross=browser.call('agent-1','append_notebook',{'text':'crossappendword','notebook':'https://wiki.test/page/'+roots['agent-2']})
        assert 'error' not in cross and cross['author']==settings['visible_labels']['agent-1']
        assert browser.db.execute('SELECT count(*) FROM audit').fetchone()[0]>before
        assert browser.db.execute('SELECT count(*) FROM request_events').fetchone()[0]>7
        assert browser.history_windows=={}
    finally:browser.close()


def test_phases_have_no_injection_preserve_exact_notes_and_reset(tmp_path):
    owner=Final512Owner();path=tmp_path/'run'
    opts={**paired_options(),'answer_final_reserve_tokens':512,'agent_log_policy':'no-agent-history-v1'}
    with patch.object(runner,'expose_log',side_effect=AssertionError('No forced log allowed')),patch.object(runner.RetainedContextClient,'before_forced_exposure',side_effect=AssertionError('No forced-exposure history manipulation allowed')):
        assert runner.run_concurrent_collaboration_pilot(path,owner,**opts)['status']=='complete'
    events=[json.loads(line) for line in (path/'evidence-index.host-only.jsonl').read_text().splitlines()]
    assert not any(e['event']=='host_metadata_log_exposure' for e in events)
    assert len([e for e in events if e['event']=='stage_published'])==9
    assert len([e for e in events if e['event']=='question_context_reset'])==2
    assert not list(path.rglob('*forced-log*'))
    assert all(len(history)==2 for history in json.loads((path/'histories.json').read_text()).values())
    for agent,history,kwargs in owner.calls:
        assert 'Host diagnostic: review the provided latest request-history' not in json.dumps(history)
    rows=json.loads((path/'results.json').read_text())
    assert len(rows)==32
    assert all(row.get('final_entry_persistence')=='exact_final_entry' for row in rows if row['phase_role']=='reflection_note')
