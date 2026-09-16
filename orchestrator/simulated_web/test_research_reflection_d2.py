"""Finite initial-research safety-cap opt-in tests."""
import json
from unittest.mock import patch

import pytest

from orchestrator.simulated_web import concurrent_research_reflection as runner
from orchestrator.simulated_web import modal_hf_titled_research_reflection as cli
from orchestrator.simulated_web.test_titled_research_reflection import options
from orchestrator.simulated_web.test_research_reflection_10c_fixes import Final512Owner
from orchestrator.simulated_web.test_same_question_pairing import paired_options


def test_d2_settings_resources_default_and_rejection():
    before,pages,_=runner.build_settings(**options())
    after,nextpages,_=runner.build_settings(**options(),preparation_safety_seconds=1200)
    assert pages==nextpages
    assert {k for k in before.keys()|after.keys() if before.get(k)!=after.get(k)}=={'policy','preparation_safety_seconds'}
    assert {k for k in before['policy'] if before['policy'][k]!=after['policy'][k]}=={'preparation_seconds'}
    assert before['policy']['preparation_seconds']==600 and after['policy']['preparation_seconds']==1200
    assert cli.resources(after)['initial_research_seconds']==1200
    assert after['policy']['preparation_generated_tokens']==8192
    assert after['policy']['answer_seconds']==180 and after['policy']['reflection_seconds']==600
    assert after['maximum_combined_generated_tokens']==54272
    for value in (True,0,601,1200.0,1800):
        with pytest.raises(ValueError,match='Preparation safety'):runner.build_settings(**options(),preparation_safety_seconds=value)


def test_runtime_1200_prompt_cap_and_conservative_admission(tmp_path):
    owner=Final512Owner();path=tmp_path/'run';original=runner.run_phase_with_readiness;seen=[]
    def spy(*args,**kwargs):
        seen.append((args[3],args[4],args[5],args[6].preparation_generated_tokens))
        return original(*args,**kwargs)
    opts={**paired_options(),'answer_final_reserve_tokens':512,'agent_log_policy':'no-agent-history-v1','notebook_title_policy':'agent-first-line-v1','preparation_safety_seconds':1200}
    with patch.object(runner,'run_phase_with_readiness',side_effect=spy):
        assert runner.run_concurrent_research_reflection(path,owner,**opts)['status']=='complete'
    prep=[r for r in seen if r[1]=='preparation']
    assert len(prep)==2 and all(r[0].startswith('Initial research: 1200 seconds.') and r[2]==1200 and r[3]==8192 for r in prep)
    assert all(r[2]==180 for r in seen if r[1]=='answer')
    assert all(r[2]==600 for r in seen if r[1]=='reflection')
    inputs=json.loads((path/'inputs.json').read_text());assert inputs['preparation_safety_seconds']==1200
    stopped=Final512Owner()
    with patch.object(runner.time,'monotonic',return_value=0):
        status=runner.run_concurrent_research_reflection(tmp_path/'short-job',stopped,**opts,job_deadline=4000)
    assert status['status']=='job_safety_stop' and stopped.calls==[]
