"""No-cloud CLI checks for explicit detached execution."""
from contextlib import nullcontext
import json
from unittest.mock import patch, MagicMock

import pytest
from orchestrator.simulated_web import modal_paired_notebook_views as cli
from orchestrator.simulated_web.test_paired_notebook_views import inputs


def test_detach_requires_explicit_launch():
    with pytest.raises(ValueError,match='requires --launch'):
        cli.main(['--run-id','mock-detached','--detach'])


@pytest.mark.parametrize('detached',[False,True])
def test_mocked_launch_propagates_detach(tmp_path,detached):
    values=inputs();args=['--run-id','mock-detached','--launch']
    fields={'dataset':'records','topic-file':'topic','editable-sources':'selectors','question-ids':'question_ids','access-manifest':'access_manifest','visible-labels':'visible_labels','round-leaders':'round_leaders'}
    for flag,key in fields.items():
        path=tmp_path/flag
        value=values[key]
        path.write_text('\n'.join(json.dumps(r) for r in value) if key=='records' else value if key=='topic' else json.dumps(value))
        args.extend(['--'+flag,str(path)])
    config=tmp_path/'modal.toml';config.write_text('[research-profile]\n')
    if detached:args.append('--detach')
    remote=MagicMock(return_value={'status':'mocked'})
    with patch.dict('os.environ',{'MODAL_PROFILE':cli.PROFILE,'MODAL_CONFIG_PATH':str(config)}),patch.object(cli.modal,'enable_output',return_value=nullcontext()),patch.object(cli.app,'run',return_value=nullcontext()) as run,patch.object(cli.execute,'remote',remote):
        cli.main(args)
    run.assert_called_once_with(detach=detached)
    remote.assert_called_once()
