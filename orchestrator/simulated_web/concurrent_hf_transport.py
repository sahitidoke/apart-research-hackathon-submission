"""Two isolated request endpoints sharing one owned vLLM process and weight copy.

Any request-level termination permanently cancels the owner and active peer HTTP
requests. Only the coordinator may warm/start the server; workers never recover it.
"""
import threading
import time

from orchestrator.simulated_web.hf_fp8 import server_command
from orchestrator.simulated_web.hf_transport import OwnedVllm


class ConcurrentOwnedVllm(OwnedVllm):
    max_num_seqs=2
    concurrent_stage_supported=True

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        server_command(self.snapshot,self.policy.context_length,self.port,max_num_seqs=2)
        self.startup_lock=threading.Lock()
        self.abort_lock=threading.Lock()
        self.active_aborts={}
        self.endpoints={}

    def _start(self,deadline):
        with self.startup_lock:
            return super()._start(deadline)

    def endpoint(self,agent):
        if agent not in ('agent-1','agent-2'):raise ValueError('Unknown agent endpoint')
        if agent not in self.endpoints:self.endpoints[agent]=AgentEndpoint(self,agent)
        return self.endpoints[agent]

    def _add_abort(self,agent,abort):
        with self.abort_lock:
            if self.cancelled.is_set():raise RuntimeError('Shared inference owner cancelled')
            if agent in self.active_aborts:raise RuntimeError('Agent already has an active request')
            token=object();self.active_aborts[agent]=(token,abort)
            return agent,token

    def _remove_abort(self,registration):
        agent,token=registration
        with self.abort_lock:
            if self.active_aborts.get(agent,(None,))[0] is token:self.active_aborts.pop(agent)

    def cancel(self):
        self.cancelled.set()
        with self.abort_lock:callbacks=[callback for _,callback in self.active_aborts.values()]
        for abort in callbacks:abort()
        super().cancel()

    def stop(self):
        self.cancel()


class AgentEndpoint:
    """Request state/events are local; ownership, immutable model and server are shared."""
    deadline_cancellation_guaranteed=True
    native_context_preflight=True

    def __init__(self,owner,agent):
        self.owner=owner;self.agent=agent
        self.events=[];self.request_lock=threading.Lock()

    def __getattr__(self,name):return getattr(self.owner,name)

    def _start(self,deadline):
        if self.owner.cancelled.is_set():raise RuntimeError('Shared inference owner cancelled')
        if not self.owner.ready or self.owner.process is None or self.owner.process.poll() is not None:
            raise RuntimeError('Shared server must be made ready by coordinator before stage admission')

    def ensure_ready(self,timeout=120):
        self._start(time.monotonic()+timeout)
        return {'status':'shared_owner_already_ready','elapsed_seconds':0,'agent_endpoint':self.agent}

    def _register_abort(self,abort):return self.owner._add_abort(self.agent,abort)
    def _unregister_abort(self,registration):self.owner._remove_abort(registration)
    def stop(self):self.owner.cancel()
    def cancel(self):self.owner.cancel()

    def count_context(self,messages,timeout=15):
        return self(self.agent,messages,timeout,num_predict=1,_count_only=True).metadata['context_preflight']

    def count_text(self,text,timeout=15):
        if not isinstance(text,str):raise ValueError('Native text counting requires a string')
        return self(self.agent,[],timeout,num_predict=1,_text_to_count=text).metadata['text_tokenization']

    def __call__(self,agent,messages,timeout,**kwargs):
        if agent!=self.agent:raise ValueError('Cross-agent request endpoint forbidden')
        if not self.request_lock.acquire(blocking=False):raise RuntimeError('Agent request already active')
        try:
            if self.owner.cancelled.is_set():raise RuntimeError('Shared inference owner cancelled')
            return OwnedVllm._request(self,agent,messages,timeout,**kwargs)
        finally:self.request_lock.release()
