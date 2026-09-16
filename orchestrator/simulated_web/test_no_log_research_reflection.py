"""10d log-removal boundaries on populated notebooks and actual reference phases."""
import json

import pytest

from orchestrator.simulated_web import modal_hf_no_log_research_reflection as cli
from orchestrator.simulated_web import concurrent_research_reflection as runner
from orchestrator.simulated_web.synchronized_notebooks import make_browser
from orchestrator.simulated_web.test_complementary_research_reflection import inputs
from orchestrator.simulated_web.test_research_reflection_10c_fixes import Final512Owner
from orchestrator.simulated_web.test_same_question_pairing import paired_options


def test_actual_input_settings_parity_and_required_optin():
    original=inputs();settings,pages,_=runner.build_settings(**original)
    changed={**original,'agent_log_policy':'no-agent-history-v1'}
    after=cli.validate_fresh('pilot-fp8-concurrent-10d',changed)
    assert {k for k in set(settings)|set(after) if settings.get(k)!=after.get(k)}=={'system_prompts','policy','agent_log_policy','log_exposure','request_history','history_search','history_search_policy'}
    assert {k for k in settings['policy'] if settings['policy'][k]!=after['policy'][k]}=={'request_history_mode'}
    assert after['policy']['request_history_mode']=='disabled'
    obsolete='Superseded automatic request-history metadata is archived; the newest payload remains. '
    for agent,prompt in settings['system_prompts'].items():
        assert obsolete in prompt
        assert after['system_prompts'][agent]==prompt.replace(obsolete,'')
    assert after['maximum_combined_generated_tokens']==54272
    assert cli.resources(after)['answer_final_reserve_tokens']==512
    with pytest.raises(ValueError,match='no agent-visible'):cli.validate_fresh('pilot-fp8-concurrent-10d',original)
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
            for url in ('https://docs.test/request-history','https://docs.test/request-history?window=old&offset=0&position=0'):
                assert 'error' in browser.call(agent,'open',{'url':url})
            browser.views.setdefault(agent,{})['old-link']=[{'label':'History','url':'https://docs.test/request-history'}]
            assert 'error' in browser.call(agent,'click',{'page_id':'old-link','link_id':1})
            assert browser.history_search_candidates(agent)==[]
            results=browser.call(agent,'search',{'query':'uniquepeerword'})['results']
            assert any(row['url'] in saved for row in results)
            assert not any('request-history' in row['url'] for row in results)
        directory=browser.call('agent-1','open',{'url':'https://wiki.test/page/'+roots['agent-2']})
        assert any('?offset=5' in link['url'] for link in directory['links'])
        page=browser.call('agent-1','open',{'url':'https://wiki.test/page/'+roots['agent-2']+'?offset=5'})
        assert any(link['url']==saved[-1] for link in page['links'])
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
    assert runner.run_concurrent_research_reflection(path,owner,**opts)['status']=='complete'
    events=[json.loads(line) for line in (path/'evidence-index.host-only.jsonl').read_text().splitlines()]
    assert not any(e['event']=='host_metadata_log_exposure' for e in events)
    assert len([e for e in events if e['event']=='stage_published'])==5
    assert len([e for e in events if e['event']=='question_context_reset'])==2
    assert not list(path.rglob('*forced-log*'))
    for agent,history,kwargs in owner.calls:
        assert 'Host diagnostic: review the provided latest request-history' not in json.dumps(history)
    rows=json.loads((path/'results.json').read_text())
    assert len(rows)==16
    assert all(row['reflection_persistence']=='exact_final_entry' for row in rows if row['phase_role']=='reflection_note')
