"""CPU-only completed-round fork/resume and six-question seed regressions."""
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
from unittest.mock import patch

import pytest

from orchestrator.simulated_web import concurrent_research_reflection as runner
from orchestrator.simulated_web import modal_hf_reference_12a as cli
from orchestrator.simulated_web.reference_resume import load_checkpoint,prepare_resume
from orchestrator.simulated_web.test_titled_research_reflection import options,inputs
from orchestrator.simulated_web.test_research_reflection_10c_fixes import Final512Owner

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'research-log/mlb-tokenpair-2026-09-12/variant-012a'
PARENT=ROOT/'runs/downloads/10d2-20260913/checkpoints/rounds-002'


def six(seed=0):
    data=options();data.update(no_peer_information=False,continuation_policy='12a-v1',model_seed=seed,preparation_safety_seconds=1200,
        question_ids=json.loads((BASE/'question-ids-same-6.host-only.json').read_text()),round_leaders=json.loads((BASE/'round-leaders-6.host-only.json').read_text()))
    return data


def test_actual_parent_validation_prefix_seed_tamper_and_no_artifacts(tmp_path):
    data=load_checkpoint(PARENT,runner.build_settings);assert data['state.json']['next_phase_index']==16
    assert prepare_resume(PARENT,six(),runner.build_settings)['extension']
    with pytest.raises(ValueError,match='seed'):prepare_resume(PARENT,six(1),runner.build_settings)
    bad=six();bad['topic']+='changed'
    with pytest.raises(ValueError,match='override'):prepare_resume(PARENT,bad,runner.build_settings)
    with pytest.raises(ValueError,match='outside'):prepare_resume(PARENT,six(),runner.build_settings,PARENT.parent.parent/'child')
    copy=tmp_path/'bad'/'checkpoints'/'rounds-002';shutil.copytree(PARENT,copy);(copy/'state.json').write_text('{}')
    with pytest.raises(ValueError,match='hash'):runner.run_concurrent_research_reflection(tmp_path/'untouched',Final512Owner(),**six(),resume_from=copy)
    assert not (tmp_path/'untouched').exists()


def test_extend_parent_then_resume_interruption_no_research_or_old_judging(tmp_path):
    before=hashlib.sha256((PARENT/'wiki.sqlite3').read_bytes()).hexdigest();owner=Final512Owner();path=tmp_path/'partial'
    def stop(checkpoint):
        if checkpoint.name=='rounds-003':raise RuntimeError('mock interruption at completed-round publication')
    with patch.object(owner,'ensure_ready',wraps=owner.ensure_ready) as ready:
        with pytest.raises(RuntimeError,match='mock interruption'):
            runner.run_concurrent_research_reflection(path,owner,**six(),resume_from=PARENT,checkpoint_callback=stop)
    assert [c.kwargs['timeout'] for c in ready.call_args_list]==[600,300]
    assert not (path/'round-00-stage-1').exists() and not (path/'round-01-stage-2').exists()
    assert (path/'inherited/round-01-stage-2/agent-1-work/phase-04.jsonl').is_file()
    rows=json.loads((path/'results.json').read_text());assert len(rows)==22 and rows[16]['global_phase_index']==16
    checkpoint=path/'checkpoints/rounds-003';assert load_checkpoint(checkpoint,runner.build_settings)['state.json']['completed_rounds']==3
    second=Final512Owner();dest=tmp_path/'continued'
    with patch.object(second,'ensure_ready',wraps=second.ensure_ready) as ready:
        status=runner.run_concurrent_research_reflection(dest,second,**six(),resume_from=checkpoint)
    assert status['status']=='complete' and status['completed_rounds']==6
    assert [c.kwargs['timeout'] for c in ready.call_args_list]==[600,300,300,300,300,300]
    assert len(second.finals)==6
    assert status['work_accounting']['inherited_phase_records']==22 and status['work_accounting']['new_phase_records']==18
    assert status['work_accounting']['new_solver_token_ceiling']==46080
    for row in json.loads((dest/'results.json').read_text())[:22]:
        assert (dest/row['log_path']).is_file()
        assert (dest/row['browser_database']).is_file()
    judged=json.loads((dest/'answer-support/summary.json').read_text())['answers']
    assert len(judged)==8 and min(r['phase_index'] for r in judged)==16
    assert status['work_accounting']['new_judge_token_ceiling']==8192
    assert hashlib.sha256((PARENT/'wiki.sqlite3').read_bytes()).hexdigest()==before
    with sqlite3.connect(dest/'wiki.sqlite3') as db:
        for author in ('agent-1','agent-2'):
            serials=[r[0] for r in db.execute('SELECT id FROM revisions WHERE agent=?',(author,))];assert len(serials)==len(set(serials))
    histories=json.loads((dest/'histories.json').read_text());assert all('answer 6 questions' in h[0]['content'] for h in histories.values())


def test_fresh_seed1_full_six_and_ceilings(tmp_path):
    owner=Final512Owner();owner.seed=1
    status=runner.run_concurrent_research_reflection(tmp_path/'fresh',owner,**six(1))
    assert status['status']=='complete' and len(owner.finals)==12
    settings=json.loads((tmp_path/'fresh/settings.json').read_text())
    assert settings['policy']['seed']==settings['model_seed']==1
    assert settings['maximum_phases']==40 and settings['maximum_combined_generated_tokens']==123904
    assert status['work_accounting']['new_solver_token_ceiling']==111616
    assert status['work_accounting']['new_judge_token_ceiling']==12288


def test_seed1_reaches_shared_transport_constructor_without_startup():
    with patch.object(cli,'ConcurrentOwnedVllm',side_effect=RuntimeError('stop before model inspection')) as factory:
        with pytest.raises(RuntimeError,match='stop before model'):
            cli.execute.get_raw_f()('mock-seed1-fp8-concurrent-12a',six(1))
    assert factory.call_args.kwargs['seed']==1
    assert factory.call_args.args[0].seed==1


def test_round6_prejudge_resume_and_completed_rejection(tmp_path):
    partial=tmp_path/'prejudge'
    def stop(checkpoint):
        if checkpoint.name=='rounds-006':raise RuntimeError('interrupt before verification')
    with pytest.raises(RuntimeError,match='interrupt before verification'):
        runner.run_concurrent_research_reflection(partial,Final512Owner(),**six(),resume_from=PARENT,checkpoint_callback=stop)
    assert not (partial/'answer-support').exists()
    owner=Final512Owner();child=tmp_path/'judged'
    original_judge=runner.judge_answers
    def require_started(*args,**kwargs):
        assert owner.ready_checks==1
        return original_judge(*args,**kwargs)
    with patch.object(owner,'ensure_ready',wraps=owner.ensure_ready) as ready, patch.object(runner,'judge_answers',side_effect=require_started):
        status=runner.run_concurrent_research_reflection(child,owner,**six(),resume_from=partial/'checkpoints/rounds-006')
    assert [c.kwargs['timeout'] for c in ready.call_args_list]==[600] and not owner.finals
    assert status['completed_rounds']==6 and status['answer_support']['answers_assessed']==8
    assert status['work_accounting']['new_phase_records']==0
    assert status['work_accounting']['new_solver_token_ceiling']==0
    assert status['work_accounting']['verification_start_index']==16
    with pytest.raises(ValueError,match='already present'):
        runner.run_concurrent_research_reflection(tmp_path/'uncreated',Final512Owner(),**six(),resume_from=child/'checkpoints/rounds-006')
    assert not (tmp_path/'uncreated').exists()
