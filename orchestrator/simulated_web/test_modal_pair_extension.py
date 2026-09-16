"""Local launcher checks only; no remote execution or model construction."""
from pathlib import Path
from unittest.mock import patch

import pytest

from orchestrator.simulated_web import modal_pair_extension as launcher


def test_selector_rejects_traversal_same_run_and_wrong_parent_boundary():
    for value in ('../x/checkpoints/rounds-003','child/checkpoints/rounds-003','old/checkpoints/rounds-002'):
        with pytest.raises(ValueError):launcher.validate_selector(value,'child',False)
    launcher.validate_selector('old/checkpoints/rounds-000','child',True)


def test_remote_manifest_mismatch_before_setup_or_owned_model(tmp_path):
    checkpoint=tmp_path/'old/checkpoints/rounds-003';checkpoint.mkdir(parents=True)
    (checkpoint/'checkpoint.json').write_text('{}')
    def local_path(value):return tmp_path if value=='/runs' else Path(value)
    with patch.object(launcher,'Path',side_effect=local_path),patch.object(launcher,'OwnedOllama') as owner:
        with pytest.warns(UserWarning,match='executing locally'),pytest.raises(ValueError,match='differs'):
            launcher.execute.local('child','old/checkpoints/rounds-003',False,'0'*64)
        owner.assert_not_called()
    assert not (tmp_path/'child').exists() and not (tmp_path/'child-setup').exists()
