"""Finite local admission checks for the explicit two-question pilot."""
import json
from unittest.mock import patch

import pytest

from orchestrator.simulated_web import modal_hf_research_reflection as cli
from orchestrator.simulated_web.test_same_question_pairing import paired_options

RUN_ID='germanwiki-2-010a-fp8-development-pilot-test-research-reflection'


def test_count_defaults_to_six_and_two_is_explicit():
    inputs=paired_options()
    with pytest.raises(ValueError,match='explicitly selected count'):cli.validate_fresh(RUN_ID,inputs)
    settings=cli.validate_fresh(RUN_ID,inputs,2)
    assert settings['question_count']==2
    assert settings['maximum_generated_tokens']==50176
    assert settings['maximum_combined_generated_tokens']==54272
    for invalid in (True,1,3,'2'):
        with pytest.raises(ValueError,match='Question count'):cli.validate_fresh(RUN_ID,inputs,invalid)
    with patch.object(cli,'build_settings',return_value=({'question_count':6,'note_retry_policy':cli.SHORT_NOTE_POLICY},[],{})):
        assert cli.validate_fresh(RUN_ID,inputs)['question_count']==6


def test_actual_cli_two_question_preflight_no_remote_actions(tmp_path,capsys):
    inputs=paired_options();argv=['--run-id',RUN_ID,'--question-count','2','--validate-only','--short-note-retry',
        '--notebook-context-policy','9e-v1','--notebook-quota-policy','notebook-exempt-v1',
        '--question-pairing-policy','same-question-v1','--source-access-policy','discovery-only-v1']
    for flag,key in [('dataset','records'),('topic-file','topic'),('editable-sources','selectors'),('question-ids','question_ids'),
                     ('access-manifest','access_manifest'),('visible-labels','visible_labels'),('round-leaders','round_leaders')]:
        p=tmp_path/flag;data=inputs[key]
        p.write_text('\n'.join(json.dumps(row) for row in data) if key=='records' else data if key=='topic' else json.dumps(data))
        argv.extend(['--'+flag,str(p)])
    with patch.object(cli.execute,'remote',side_effect=AssertionError('No remote calls')),patch.object(cli.subprocess,'run',side_effect=AssertionError('No downloads')):
        cli.main(argv)
    output=json.loads(capsys.readouterr().out)
    assert output['status']=='validated_no_cloud_actions'
    assert output['run_scope']=={'purpose':'two-question development pilot','question_count':2}
    assert output['resources']['phase_records']==16 and output['resources']['normal_model_phases']==12
    assert output['resources']['hard_job_seconds']==21600 and output['resources']['gpu_count']==1
