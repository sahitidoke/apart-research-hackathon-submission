"""Finite host-only partition and browser discovery regressions; no inference."""
from copy import deepcopy
import hashlib
import json

import pytest

from orchestrator.simulated_web.browser import Browser
from orchestrator.simulated_web import modal_hf_complementary_collaboration_pilot as cli
from orchestrator.simulated_web.complementary_partition import build_partition, identity, main
from orchestrator.simulated_web.source_access import access_plan, access_browser_options
from orchestrator.simulated_web.source_discovery import discovery_plan, browser_discovery


def fixture():
    records = []
    for n in range(2):
        paragraphs = [{'idx': i, 'title': title, 'paragraph_text': body, 'is_supporting': True}
                      for i,(title,body) in enumerate([('orientation','first fact'), ('bridge','bridge fact '+str(n)), ('league','league fact'), ('terminal'+str(n),'terminal fact '+str(n))])]
        records.append({'id': 'q'+str(n), 'paragraphs': paragraphs, 'question_decomposition': [{'paragraph_support_idx': i} for i in range(4)]})
    return records, {a: ['q0','q1'] for a in ('agent-1','agent-2')}


def test_exact_duplicates_families_complementarity_terminal_roles_and_reproducibility():
    records, questions = fixture()
    manifest = build_partition(records, questions)
    assert manifest == build_partition(records, questions)
    audit = manifest['partition_audit']
    assert audit['objective']['exact_support_optimum'][:2] == [0,2]
    assert len(audit['duplicate_groups']) == 2
    owners = {g:v['owner'] for g,v in audit['groups'].items()}
    assert owners[identity(records[0]['paragraphs'][1])] == owners[identity(records[1]['paragraphs'][1])]
    assert {v['terminal_owner'] for v in audit['questions'].values()} == {'agent-1','agent-2'}
    assert all(min(v['support_counts'].values()) > 0 for v in audit['questions'].values())
    assert set(manifest['source_groups']['agent-1']).isdisjoint(manifest['source_groups']['agent-2'])


def test_equal_body_different_title_is_colocated():
    records, questions = fixture()
    records[1]['paragraphs'][1]['title'] = 'other bridge title'
    records[1]['paragraphs'][1]['paragraph_text'] = records[0]['paragraphs'][1]['paragraph_text']
    audit = build_partition(records, questions)['partition_audit']
    assert audit['groups'][identity(records[0]['paragraphs'][1])]['owner'] == audit['groups'][identity(records[1]['paragraphs'][1])]['owner']


def test_no_artifacts_for_invalid_or_infeasible_input(tmp_path):
    records, questions = fixture()
    for row in records:
        for p in row['paragraphs']:p['title']='one family'
    data, ids = tmp_path/'data', tmp_path/'ids'
    data.write_text('\n'.join(json.dumps(r) for r in records));ids.write_text(json.dumps(questions))
    output=tmp_path/'untouched'/'manifest.json'
    with pytest.raises(ValueError, match='No complementary'):
        main(['--dataset',str(data),'--question-ids',str(ids),'--output',str(output)])
    assert not output.parent.exists()
    with pytest.raises(ValueError, match='Unknown'):
        build_partition(records, {a:['missing','q0'] for a in questions})


def test_discovery_listing_filter_foreign_get_and_manifest_tamper():
    records, questions = fixture();manifest=build_partition(records,questions)
    pages=[]
    for row in records:
        prefix='https://docs.test/q/'+hashlib.sha256(row['id'].encode()).hexdigest()+'/'
        links=[]
        for i,p in enumerate(row['paragraphs']):
            url=f'{prefix}p/{i}/0';links.append({'url':url,'label':p['title']})
            pages.append({'url':url,'title':p['title'],'text':p['paragraph_text'],'links':[]})
        pages.append({'url':prefix,'title':'Collection','text':'Available documents:\n','links':links})
    pages.append({'url':'https://docs.test/','title':'Root','text':'Available documents:\n','links':[{'url':p['url'],'label':p['title']} for p in pages if '/p/' in p['url']]})
    discovery=discovery_plan(records,pages,[])
    plan=access_plan(records,pages,discovery,manifest,True,'discovery_only')
    browser=Browser(pages,':memory:',**browser_discovery(discovery),**access_browser_options(plan))
    try:
        for agent in questions:
            foreign=set().union(*(set(urls) for g,urls in discovery['groups'].items() if g not in manifest['source_groups'][agent]))
            assert foreign
            for url in discovery['listing_urls']:
                assert not foreign & {link['url'] for link in browser.call(agent,'open',{'url':url})['links']}
            assert not foreign & {row['url'] for row in browser.search('fact',agent)['results']}
            assert 'error' not in browser.call(agent,'open',{'url':sorted(foreign)[0]})
    finally:browser.close()
    altered=deepcopy(manifest);altered['source_groups']['agent-1'].append(altered['source_groups']['agent-2'][0])
    with pytest.raises(ValueError,match='differs'):access_plan(records,pages,discovery,altered)
    altered=deepcopy(manifest);altered['dataset_sha256']='bad'
    with pytest.raises(ValueError,match='Invalid'):access_plan(records,pages,discovery,altered)


def test_11b_entrypoint_rejects_legacy_partition_before_cloud():
    with pytest.raises(ValueError, match='audited complementary'):
        cli.validate_fresh('pilot-fp8-concurrent-11b', {'inference_profile':'hf-fp8-v1','access_manifest':{'schema':'source-access-v1'}})
    with pytest.raises(ValueError, match='end in'):
        cli.validate_fresh('pilot-fp8-concurrent-11a', {})
