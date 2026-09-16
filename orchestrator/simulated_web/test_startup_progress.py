"""Logging-only readiness tests with real lightweight heartbeat threads."""
import threading
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from orchestrator.simulated_web.concurrent_collaboration_pilot import readiness_with_progress


@pytest.mark.parametrize('fail',[False,True])
def test_readiness_progress_flush_ownership_single_call_and_stopped_thread(fail):
    caller=threading.get_ident();waiting=threading.Event();lines=[];calls=[]
    def output(*args,**kwargs):
        assert kwargs.get('flush') is True
        line=' '.join(map(str,args));lines.append(line)
        if 'readiness waiting;' in line:waiting.set()
    def ready(*,timeout):
        assert threading.get_ident()==caller
        calls.append(timeout)
        assert 'readiness starting;' in lines[0]
        assert 'timeout=600s' in lines[0] and 'model log=/runs/setup/vllm.log' in lines[0]
        assert waiting.wait(2)
        if fail:raise RuntimeError('private exception contents must not be printed')
        return {'status':'mock_ready'}
    owner=SimpleNamespace(ensure_ready=ready,log_path='/runs/setup/vllm.log')
    with patch('builtins.print',side_effect=output):
        if fail:
            with pytest.raises(RuntimeError,match='private exception'):
                readiness_with_progress(owner,600,'Initial startup',interval_seconds=0.01)
        else:
            assert readiness_with_progress(owner,600,'Initial startup',interval_seconds=0.01)=={'status':'mock_ready'}
    assert calls==[600]
    assert any('elapsed=' in line and 'remaining=' in line for line in lines)
    assert any(('readiness failed;' if fail else 'readiness ready;') in line for line in lines)
    assert not any('private exception contents' in line for line in lines)
    assert not any(t.name=='collaboration-readiness-progress' for t in threading.enumerate())


def test_immediate_readiness_unchanged_timeout_and_no_model_log(capsys):
    owner=SimpleNamespace(ensure_ready=Mock(return_value={'ready':True}))
    assert readiness_with_progress(owner,300,'Later stage')=={'ready':True}
    owner.ensure_ready.assert_called_once_with(timeout=300)
    text=capsys.readouterr().out
    assert 'model log=unavailable' in text and 'readiness ready;' in text
    assert 'readiness waiting;' not in text
    assert not any(t.name=='collaboration-readiness-progress' for t in threading.enumerate())
