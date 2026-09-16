"""Finite mock-only FP8 identity, matched settings and checkpoint regressions."""
import json
from unittest.mock import patch

import pytest

from orchestrator.simulated_web import modal_hf_synchronized_exchange as cli
from orchestrator.simulated_web.hf_fp8 import MODEL, REVISION, VLLM_VERSION, KV_CACHE_DTYPE
from orchestrator.simulated_web.synchronized_exchange import build_settings, run_exchange, load_checkpoint, SHORT_NOTE_POLICY
from orchestrator.simulated_web.test_synchronized_exchange import inputs, Client


class FP8Client(Client):
    def inspect_model(self, timeout=30):
        return {'name': MODEL, 'revision': REVISION, 'digest': 'hf:' + REVISION,
                'backend': 'vllm', 'backend_version': VLLM_VERSION,
                'kv_cache_dtype': KV_CACHE_DTYPE, 'all_artifact_checksums_verified': True}


def test_profile_preserves_protocol_and_no_peer_is_only_treatment():
    legacy = build_settings(**inputs(), note_retry_policy=SHORT_NOTE_POLICY)[0]
    fp8 = build_settings(**inputs(), note_retry_policy=SHORT_NOTE_POLICY, inference_profile='hf-fp8-v1')[0]
    assert fp8.pop('inference_profile') == 'hf-fp8-v1'
    assert fp8 == legacy
    treatment = build_settings(**inputs(), note_retry_policy=SHORT_NOTE_POLICY,
                               inference_profile='hf-fp8-v1', no_peer_information=True)[0]
    assert treatment.pop('no_peer_information') is True
    assert treatment.pop('inference_profile') == 'hf-fp8-v1'
    assert treatment == legacy


def test_identity_rejected_before_output_and_no_legacy_fallback(tmp_path):
    with pytest.raises(ValueError, match='FP8 model contract'):
        run_exchange(tmp_path/'wrong', Client(), **inputs(), inference_profile='hf-fp8-v1')
    assert not (tmp_path/'wrong').exists()
    with pytest.raises(ValueError, match='pinned model digest'):
        run_exchange(tmp_path/'legacy', FP8Client(), **inputs())
    assert not (tmp_path/'legacy').exists()
    with pytest.raises(ValueError, match='Unknown paired inference'):
        build_settings(**inputs(), inference_profile='unapproved')


@pytest.mark.parametrize('no_peer', [False, True])
def test_fp8_protocol_complete_checkpoint_and_resume(tmp_path, no_peer):
    def stop(checkpoint):
        if checkpoint.name == 'rounds-001':
            raise RuntimeError('mock interruption')
    with pytest.raises(RuntimeError, match='mock interruption'):
        run_exchange(tmp_path/'first', FP8Client(), **inputs(), inference_profile='hf-fp8-v1',
                     note_retry_policy=SHORT_NOTE_POLICY, no_peer_information=no_peer,
                     checkpoint_callback=stop)
    checkpoint=tmp_path/'first/checkpoints/rounds-001'
    saved, browser, _ = load_checkpoint(checkpoint)
    browser.close()
    assert saved['settings.json']['inference_profile'] == 'hf-fp8-v1'
    assert saved['settings.json']['provenance']['model']['digest'] == 'hf:' + REVISION
    assert run_exchange(tmp_path/'resumed', FP8Client(), resume_from=checkpoint)['status'] == 'complete'
    final=json.loads((tmp_path/'resumed/settings.json').read_text())
    assert final.get('no_peer_information', False) == no_peer
    assert final['policy']['context_length'] == 65536
    assert final['note_attempt_token_limits'] == [512,1024]


def test_cli_validation_has_no_cloud_or_download(tmp_path, capsys):
    value=inputs()
    # Unique fixture questions cannot represent six rounds; settings builder is
    # replaced only at the CLI boundary, while real build/run is tested above.
    argv=['--run-id','mock-fp8-matched','--short-note-retry','--validate-only']
    for flag,field in [('dataset','records'),('topic-file','topic'),('editable-sources','selectors'),
                       ('question-ids','question_ids'),('access-manifest','access_manifest'),
                       ('visible-labels','visible_labels'),('round-leaders','round_leaders')]:
        path=tmp_path/flag
        data=value[field]
        path.write_text('\n'.join(json.dumps(row) for row in data) if field=='records' else
                        data if field=='topic' else json.dumps(data))
        argv.extend(['--'+flag,str(path)])
    with patch.object(cli, 'validate_fresh', return_value={'question_count':6}) as validate, \
         patch.object(cli.execute, 'remote', side_effect=AssertionError('cloud forbidden')), \
         patch.object(cli.subprocess, 'run', side_effect=AssertionError('download forbidden')):
        cli.main(argv)
    assert validate.call_args.args[1]['inference_profile']=='hf-fp8-v1'
    report=json.loads(capsys.readouterr().out)
    assert report['resources']['maximum_generated_tokens']==122880
    assert report['resources']['hard_job_seconds']==21600
    assert report['resources']['gpu_count']==1
    with pytest.raises(ValueError,match='require --launch'):
        cli.main(argv+['--download-model'])
