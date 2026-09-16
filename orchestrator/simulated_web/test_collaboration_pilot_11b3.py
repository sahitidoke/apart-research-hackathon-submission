"""Finite cold-start readiness tests; no inference or cloud execution."""
import json
from unittest.mock import patch

import pytest

from orchestrator.simulated_web import concurrent_collaboration_pilot as runner
from orchestrator.simulated_web.test_collaboration_pilot_11b2 import Final512Owner
from orchestrator.simulated_web.test_same_question_pairing import paired_options


OPTIONS={'answer_final_reserve_tokens':512,'coordinator_readiness_policy':'cold-start-600-v2'}


def test_startup600_only_first_coordinator_preserves_worker_and_phase_policies(tmp_path):
    before,pages,_=runner.build_settings(**paired_options(),answer_final_reserve_tokens=512)
    after,next_pages,_=runner.build_settings(**paired_options(),**OPTIONS)
    assert pages==next_pages
    assert {k for k in set(before)|set(after) if before.get(k)!=after.get(k)}=={'coordinator_readiness'}
    assert 'coordinator_readiness' not in before
    assert after['policy']==before['policy']
    assert after['policy']['initial_readiness_timeout_seconds']==300
    assert after['policy']['readiness_timeout_seconds']==120
    assert after['policy']['answer_seconds']==180 and after['maximum_combined_generated_tokens']==54272
    owner=Final512Owner();path=tmp_path/'new'
    with patch.object(owner,'ensure_ready',wraps=owner.ensure_ready) as ready:
        assert runner.run_concurrent_collaboration_pilot(path,owner,**paired_options(),**OPTIONS)['status']=='complete'
    assert [call.kwargs['timeout'] for call in ready.call_args_list]==[600]+[300]*8
    assert json.loads((path/'round-00-stage-1/shared-readiness.json').read_text())['timeout_seconds']==600
    assert json.loads((path/'round-01-stage-2/shared-readiness.json').read_text())['timeout_seconds']==300
    owner=Final512Owner()
    with patch.object(owner,'ensure_ready',wraps=owner.ensure_ready) as ready:
        assert runner.run_concurrent_collaboration_pilot(tmp_path/'historical',owner,**paired_options(),answer_final_reserve_tokens=512)['status']=='complete'
    assert [call.kwargs['timeout'] for call in ready.call_args_list]==[300]*9


def test_coldstartup_failure_preserves_artifact_before_any_phase(tmp_path):
    owner=Final512Owner();path=tmp_path/'failure'
    def fail(timeout):
        assert timeout==600
        assert json.loads((path/'round-00-stage-1/shared-readiness.json').read_text())['status']=='in_progress'
        raise RuntimeError('synthetic startup budget exhausted')
    with patch.object(owner,'ensure_ready',side_effect=fail):
        with pytest.raises(RuntimeError,match='synthetic startup'):
            runner.run_concurrent_collaboration_pilot(path,owner,**paired_options(),**OPTIONS)
    report=json.loads((path/'round-00-stage-1/shared-readiness.json').read_text())
    assert report['status']=='failed' and report['timeout_seconds']==600
    assert json.loads((path/'results.json').read_text())==[]
    status=json.loads((path/'manifest.json').read_text())
    assert status['status']=='failed' and not status['prepared'] and status['completed_rounds']==0


def test_initial_admission_adds300_without_changing_later_or_phase_caps(tmp_path):
    owner=Final512Owner();path=tmp_path/'short-job'
    # Legacy initial threshold2880; opt-in threshold3180.3100 must now decline.
    with patch.object(runner.time,'monotonic',return_value=0),patch.object(owner,'ensure_ready',wraps=owner.ensure_ready) as ready:
        result=runner.run_concurrent_collaboration_pilot(path,owner,**paired_options(),**OPTIONS,job_deadline=3100)
    assert result['solver_status']=='job_safety_stop' and not result['prepared']
    ready.assert_not_called()
    assert not (path/'round-00-stage-1').exists()
    with pytest.raises(ValueError):runner.build_settings(**paired_options(),coordinator_readiness_policy='unknown')
