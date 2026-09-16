"""Finite safety-v3 cap and narrowly permitted completed-round migration checks."""
import json
from unittest.mock import patch

import pytest

from orchestrator.simulated_web import concurrent_research_reflection as runner
from orchestrator.simulated_web.reference_ablations import ARMS
from orchestrator.simulated_web.reference_resume import prepare_resume,load_checkpoint
from orchestrator.simulated_web.test_reference_12a import six,PARENT
from orchestrator.simulated_web.test_research_reflection_10c_fixes import Final512Owner


@pytest.mark.parametrize('arm',ARMS)
def test_only_answer_time_changes_and_old_defaults_remain(arm):
    old,pages,_=runner.build_settings(**six(1),ablation_policy=arm)
    new,other,_=runner.build_settings(**six(1),ablation_policy=arm,answer_safety_seconds=600)
    assert old['policy']['answer_seconds']==180 and new['policy']['answer_seconds']==600
    assert {k for k in old['policy'] if old['policy'][k]!=new['policy'][k]}=={'answer_seconds'}
    assert {k for k in old.keys()|new.keys() if old.get(k)!=new.get(k)}=={'policy','answer_timing_policy','answer_safety_seconds'}
    assert old['maximum_combined_generated_tokens']==new['maximum_combined_generated_tokens']==123904
    assert old['system_prompts']==new['system_prompts'] and pages==other


@pytest.mark.parametrize('value',[600,0,601,True])
def test_baseline12a_rejects_answer_safety_override(value):
    with pytest.raises(ValueError):runner.build_settings(**six(),answer_safety_seconds=value)
    assert runner.build_settings(**six())[0]['policy']['answer_seconds']==180


@pytest.mark.parametrize('arm',ARMS)
def test_same_arm_migration_runtime_cap_resume_and_downgrade_rejection(tmp_path,arm):
    old={**six(1),'ablation_policy':arm};new={**old,'answer_safety_seconds':600}
    def stop_at(round_index):
        def stop(cp):
            if cp.name==f'rounds-{round_index:03d}':raise RuntimeError('bounded checkpoint interruption')
        return stop
    first=tmp_path/'old'
    with pytest.raises(RuntimeError,match='bounded checkpoint'):
        runner.run_concurrent_research_reflection(first,Final512Owner(),**old,checkpoint_callback=stop_at(1))
    cp=first/'checkpoints/rounds-001'
    checked=prepare_resume(cp,new,runner.build_settings)
    assert checked['answer_timing_migration']['inherited_answer_seconds']==180
    assert load_checkpoint(cp,runner.build_settings)['settings.json']['policy']['answer_seconds']==180
    for changed in [{**new,'model_seed':0},{**new,'ablation_policy':next(p for p in ARMS if p!=arm)},{**new,'topic':'changed'}]:
        with pytest.raises(ValueError):prepare_resume(cp,changed,runner.build_settings)
    child=tmp_path/'new'
    with patch.object(runner,'run_phase_with_readiness',wraps=runner.run_phase_with_readiness) as phases:
        with pytest.raises(RuntimeError,match='bounded checkpoint'):
            runner.run_concurrent_research_reflection(child,Final512Owner(),**new,resume_from=cp,checkpoint_callback=stop_at(2))
    calls=[c for c in phases.call_args_list if c.args[4]=='answer']
    assert len(calls)==2 and all(c.args[5]==600 and 'Answer phase: 600 seconds.' in c.args[3] for c in calls)
    rows=json.loads((child/'results.json').read_text())
    answers=[r for r in rows if r['phase_role']=='answer']
    assert [r['budget_seconds'] for r in answers]==[180,180,600,600]
    assert all(r['generated_token_allowance']==2048 and r['final_reserve_tokens']==512 for r in answers)
    migration=json.loads((child/'settings.json').read_text())['resume']['answer_timing_migration']
    assert migration['boundary_completed_rounds']==1 and migration['new_answer_seconds']==600
    next_cp=child/'checkpoints/rounds-002'
    assert prepare_resume(next_cp,new,runner.build_settings)['answer_timing_migration'] is None
    with pytest.raises(ValueError,match='override'):prepare_resume(next_cp,old,runner.build_settings)
    if arm=='generic-titles-v1':
        third=tmp_path/'again'
        with pytest.raises(RuntimeError,match='bounded checkpoint'):
            runner.run_concurrent_research_reflection(third,Final512Owner(),**new,resume_from=next_cp,checkpoint_callback=stop_at(3))
        assert json.loads((third/'settings.json').read_text())['resume']['answer_timing_migration']==migration
    with pytest.raises(ValueError):prepare_resume(PARENT,{**six(), 'answer_safety_seconds':600},runner.build_settings)
