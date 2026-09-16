"""Finite actual-thread cleanup and answer-only reserve regressions; no model runs."""
import json
from pathlib import Path
import sqlite3
import threading
from unittest.mock import patch

import pytest

from orchestrator.simulated_web import concurrent_research_reflection as runner
from orchestrator.simulated_web import modal_hf_complementary_research_reflection as cli
from orchestrator.simulated_web.runner import ModelResponse
from orchestrator.simulated_web.test_concurrent_research_reflection import Owner as MockOwner, Endpoint as MockEndpoint
from orchestrator.simulated_web.test_same_question_pairing import paired_options


REPLAY=json.loads((Path(__file__).with_name('fixtures')/'11b-three-citation-replay.host-only.json').read_text())


class Final512Endpoint(MockEndpoint):
    def __call__(self, agent, history, timeout, **kwargs):
        if kwargs.get('format_schema'):
            assert kwargs['final_only'] and kwargs['num_predict'] in (256,512)
            self.owner.finals.append((agent, kwargs))
            if kwargs['num_predict']==256:
                return ModelResponse({'content':REPLAY['observed_final256']},{'eval_count':256,'prompt_eval_count':100,'done_reason':'length'})
            return ModelResponse({'content':REPLAY['full_draft']},{'eval_count':REPLAY['synthetic_full_final_tokens'],'prompt_eval_count':100,'done_reason':'stop'})
        return super().__call__(agent, history, timeout, **kwargs)


class Final512Owner(MockOwner):
    def __init__(self):super().__init__();self.finals=[]
    def endpoint(self, agent):
        if agent not in self.endpoints:self.endpoints[agent]=Final512Endpoint(self,agent)
        return self.endpoints[agent]


def test_answer_final512_within2048_research_unchanged_and_historical_default(tmp_path):
    before, pages, _=runner.build_settings(**paired_options())
    after, next_pages, _=runner.build_settings(**paired_options(), answer_final_reserve_tokens=512)
    assert next_pages == pages
    assert {k for k in set(before)|set(after) if before.get(k)!=after.get(k)} == {'answer_final_reserve_tokens'}
    assert 'answer_final_reserve_tokens' not in before
    assert before['policy']['final_reserve_tokens']==after['policy']['final_reserve_tokens']==256
    assert after['maximum_combined_generated_tokens']==54272 and after['policy']['answer_seconds']==180
    assert cli.resources(after)['answer_final_reserve_tokens']==512
    owner=Final512Owner();path=tmp_path/'run'
    assert runner.run_concurrent_research_reflection(path,owner,**paired_options(),answer_final_reserve_tokens=512)['status']=='complete'
    assert len(owner.finals)==4
    assert all(call['num_predict']==512 for _,call in owner.finals)
    results=json.loads((path/'results.json').read_text())
    answers=[r for r in results if r['phase_role']=='answer']
    assert all(r['final_reserve_tokens']==512 and r['generated_tokens_observed'] <= 2048 for r in answers)
    assert all(r['final_reserve_tokens']==0 for r in results if r['phase_role'] in ('initial_research','reflection'))
    assert all(len(json.loads(r['raw_answer_json'])['citations'])==3 for r in answers)
    assert all(r['final_attempted'] and r['answer_format_enforcement']=='schema_and_host' for r in answers)
    with pytest.raises(ValueError):runner.build_settings(**paired_options(),answer_final_reserve_tokens=1024)


class SimulatedInputCancellation(BaseException):
    pass


class CancellationEndpoint(MockEndpoint):
    def __call__(self, agent, history, timeout, **kwargs):
        with self.owner.lock:
            self.owner.active += 1
            if self.owner.active == 2:self.owner.both_started.set()
        try:
            assert self.owner.cancelled.wait(5)
            raise InterruptedError('worker request interrupted after coordinator cancellation')
        finally:
            with self.owner.lock:self.owner.active -= 1


class CancellationOwner(MockOwner):
    def __init__(self):super().__init__();self.both_started=threading.Event()
    def endpoint(self, agent):
        if agent not in self.endpoints:self.endpoints[agent]=CancellationEndpoint(self,agent)
        return self.endpoints[agent]


def test_coordinator_cancellation_settles_and_closes_real_sqlite_in_owning_threads(tmp_path):
    owner=CancellationOwner();opened=[];closed=[];original_connect=sqlite3.connect
    class TrackedConnection(sqlite3.Connection):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs);self.creator=threading.get_ident();opened.append(self)
        def close(self):
            assert threading.get_ident()==self.creator
            super().close();closed.append(self)
    def connect(*args,**kwargs):return original_connect(*args,**kwargs,factory=TrackedConnection)
    def interrupted(_futures):
        assert owner.both_started.wait(5)
        raise SimulatedInputCancellation('synthetic coordinator input cancellation')
    path=tmp_path/'cancelled'
    with patch.object(runner.sqlite3,'connect',side_effect=connect),patch.object(runner,'as_completed',side_effect=interrupted):
        with pytest.raises(SimulatedInputCancellation,match='synthetic coordinator'):
            runner.run_concurrent_research_reflection(path,owner,**paired_options())
    assert owner.active==0 and owner.cancelled.is_set()
    assert {id(c) for c in opened}=={id(c) for c in closed}
    assert len({c.creator for c in opened}) >= 3
    status=json.loads((path/'manifest.json').read_text())
    assert status['status']=='failed' and status['error'].startswith('SimulatedInputCancellation:')
    assert not (path/'round-00-stage-1/barrier.json').exists()
    for agent in ('agent-1','agent-2'):
        work=path/f'round-00-stage-1/{agent}-work'
        assert json.loads((work/'worker-status.json').read_text())['status']=='failed'
        assert (work/'results.json').is_file() and (work/'history.json').is_file()


def test_cancel_cleanup_error_does_not_replace_original_cancellation(tmp_path):
    owner=CancellationOwner();normal_cancel=owner.cancel
    def failing_cancel():normal_cancel();raise RuntimeError('secondary cleanup failure')
    def interrupted(_futures):
        assert owner.both_started.wait(5)
        raise SimulatedInputCancellation('original signal')
    with patch.object(owner,'cancel',side_effect=failing_cancel),patch.object(runner,'as_completed',side_effect=interrupted):
        with pytest.raises(SimulatedInputCancellation,match='original signal'):
            runner.run_concurrent_research_reflection(tmp_path/'cancelled',owner,**paired_options())
    status=json.loads((tmp_path/'cancelled/manifest.json').read_text())
    assert 'original signal' in status['error'] and 'secondary cleanup failure' in status['cancellation_error']


def test_observed_three_citation_truncation_replays_failure_at256(tmp_path):
    owner=Final512Owner();path=tmp_path/'old-reserve'
    with pytest.raises(RuntimeError,match='Concurrent stage failed'):
        runner.run_concurrent_research_reflection(path,owner,**paired_options())
    rows=json.loads((path/'results.json').read_text())
    answers=[r for r in rows if r['phase_role']=='answer']
    assert answers and all(r['status']=='invalid_final_response' for r in answers)
    assert all(r['final_reserve_tokens']==256 for r in answers)
