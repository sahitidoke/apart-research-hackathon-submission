"""Mock-only opt-in source-denial wording and checkpoint provenance contracts."""
import hashlib
import json
from unittest.mock import patch

import pytest

from orchestrator.simulated_web import modal_synchronized_exchange as cli
from orchestrator.simulated_web.synchronized_exchange import build_settings, load_checkpoint, run_exchange
from orchestrator.simulated_web.synchronized_notebooks import SOURCE_DENIAL_MESSAGE, SOURCE_DENIAL_POLICY, make_browser, fork_stage
from orchestrator.simulated_web.test_synchronized_exchange import Client, inputs


@pytest.mark.parametrize('enabled',[False,True])
def test_denied_documents_routes_audit_and_snapshot(tmp_path,enabled):
    settings,pages,_=build_settings(**inputs(),source_denial_policy=SOURCE_DENIAL_POLICY if enabled else None)
    browser=make_browser(settings,pages,':memory:')
    snapshots={}
    try:
        agent='agent-1'
        denied=next(url for url in browser.pages if url not in browser.access_allowed_urls[agent])
        allowed=next(url for url in browser.pages if url in browser.access_allowed_urls[agent])
        expected=SOURCE_DENIAL_MESSAGE if enabled else 'Access denied'
        for _ in range(2):assert browser.call(agent,'open',{'url':denied})=={'error':expected}
        assert 'error' not in browser.call(agent,'open',{'url':allowed})
        rendered=browser.render(agent,allowed,'Links','', [{'label':'Document','url':denied}])
        assert browser.call(agent,'click',{'page_id':rendered['page_id'],'link_id':1})=={'error':expected}
        audit=json.loads(browser.db.execute('SELECT result FROM audit ORDER BY id DESC LIMIT 1').fetchone()[0])
        assert audit=={'error':expected}
        assert browser.db.execute('SELECT status FROM request_events ORDER BY id DESC LIMIT 1').fetchone()[0]=='error'
        assert browser.call(agent,'open',{'url':'https://docs.test/nonexistent'})=={'error':'Access denied'}
        snapshots=fork_stage(browser,settings,pages,tmp_path,1,1)
        assert snapshots[agent].call(agent,'open',{'url':denied})=={'error':expected}
        if enabled:
            assert settings['source_denial_message']==expected
        else:
            assert 'source_denial_policy' not in settings and 'source_denial_message' not in settings
    finally:
        for snapshot in snapshots.values():snapshot.close()
        browser.close()


def test_policy_validation_precedes_output(tmp_path):
    with pytest.raises(ValueError,match='Unknown source denial policy'):
        run_exchange(tmp_path/'bad',Client(),**inputs(),source_denial_policy='unknown')
    assert not (tmp_path/'bad').exists()
    with pytest.raises(ValueError,match='Resume overrides prohibited'):
        run_exchange(tmp_path/'override',Client(),resume_from=tmp_path/'unused',source_denial_policy=SOURCE_DENIAL_POLICY)
    assert not (tmp_path/'override').exists()


def test_checkpoint_preserves_policy_and_rejects_changed_wording(tmp_path):
    path=tmp_path/'original'
    def stop(checkpoint):
        if checkpoint.name=='rounds-001':raise RuntimeError('mock stop')
    with pytest.raises(RuntimeError,match='mock stop'):
        run_exchange(path,Client(),**inputs(),source_denial_policy=SOURCE_DENIAL_POLICY,checkpoint_callback=stop)
    checkpoint=path/'checkpoints/rounds-001'
    data,browser,_=load_checkpoint(checkpoint)
    assert data['settings.json']['source_denial_message']==SOURCE_DENIAL_MESSAGE
    assert browser.source_denial_policy==SOURCE_DENIAL_POLICY
    browser.close()
    assert run_exchange(tmp_path/'resume',Client(),resume_from=checkpoint)['status']=='complete'
    data,browser,_=load_checkpoint(tmp_path/'resume/checkpoints/rounds-002')
    assert browser.source_denial_policy==SOURCE_DENIAL_POLICY
    browser.close()
    settings=checkpoint/'settings.json'
    changed=json.loads(settings.read_text());changed['source_denial_message']='Different notice'
    settings.write_text(json.dumps(changed))
    manifest=checkpoint/'synchronized-checkpoint.json'
    changed=json.loads(manifest.read_text());changed['files_sha256']['settings.json']=hashlib.sha256(settings.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(changed))
    with pytest.raises(ValueError,match='Checkpoint settings mismatch'):load_checkpoint(checkpoint)


def test_cli_opt_in_validation_and_resume_override(tmp_path,capsys):
    args=['--run-id','mock-denial','--clear-source-denial']
    values=inputs()
    for flag,key in {'dataset':'records','topic-file':'topic','editable-sources':'selectors','question-ids':'question_ids','access-manifest':'access_manifest','visible-labels':'visible_labels','round-leaders':'round_leaders'}.items():
        path=tmp_path/flag;value=values[key]
        path.write_text('\n'.join(json.dumps(r) for r in value) if key=='records' else value if key=='topic' else json.dumps(value))
        args.extend(['--'+flag,str(path)])
    config=tmp_path/'modal.toml';config.write_text('[research-profile]\n')
    with patch.dict('os.environ',{'MODAL_PROFILE':cli.PROFILE,'MODAL_CONFIG_PATH':str(config)}),patch.object(cli.execute,'remote') as remote:
        cli.main(args)
    remote.assert_not_called()
    result=json.loads(capsys.readouterr().out)
    assert result['status']=='validated_no_cloud_actions'
    assert result['settings']['source_denial_policy']==SOURCE_DENIAL_POLICY
    assert result['settings']['source_denial_message']==SOURCE_DENIAL_MESSAGE
    with pytest.raises(ValueError,match='Resume overrides prohibited'):
        cli.main(['--run-id','mock-denial','--clear-source-denial','--resume-from','unused'])
