"""Transparent browser observation retention; no model summaries or history resets."""
import hashlib
import json
import time

from orchestrator.simulated_web.browser_retention import BROWSER_TOOLS
from orchestrator.simulated_web.notebook_tools import NAMES as NOTEBOOK_TOOL_NAMES
from orchestrator.simulated_web.musique_batch import write_json
from orchestrator.simulated_web.timed_transport import ContextExhausted

MARKER = '[Browser response omitted by bounded context policy. Its original source URL remains in the preceding request; reopen it if needed. Raw evidence is preserved by the host.]'
NOTICE = ('Your own messages, questions, answers and notebook updates remain in context; there is no full conversation reset. '
          'Only the newest automatically provided request-history payload is retained. If native input-token pressure requires it, '
          'older browser response bodies from completed questions are omitted, oldest first. Current-question evidence and your '
          'own text remain available. Omitted sources can be reopened through their original URLs. Raw observations remain in host artifacts. '
          'No generated summaries substitute for omitted evidence.')


def settings_contract():
    return {'policy':'bounded_browser_v1','prompt_target_fraction':0.75,'request_template_reserve_tokens':1024,
            'forced_history':'newest_payload_only','browser_observations':'oldest_completed_question_first_on_native_token_pressure',
            'protected':'system, user, assistant messages and tool requests; current-question browser responses; newest forced history',
            'failure':'stop if protected input cannot fit; no model summary or full reset'}


class BoundedContextClient:
    """Fit the actual mutable history before every model request, including note retries."""
    def __init__(self,client,policy,run_dir):
        self.client=client;self.policy=policy;self.run_dir=run_dir
        self.starts={};self.serial=0

    def __getattr__(self,name):return getattr(self.client,name)

    def begin_question(self,agent,history):
        self.starts[agent]=len(history)

    def archive_mask(self,agent,history,indices,reason,diagnostics=None):
        if not indices:return
        self.serial+=1
        stem=f'bounded-context-{self.serial:04d}-{agent}'
        before=json.loads(json.dumps(history))
        snapshot=stem+'-before.json'
        report={'policy':'bounded_browser_v1','agent':agent,'reason':reason,'history_snapshot':snapshot,
                'masked_message_indices':indices,'original_content_sha256':{
                    str(i):hashlib.sha256(history[i]['content'].encode()).hexdigest() for i in indices},
                'removed_content_characters':sum(len(history[i]['content']) for i in indices),
                'diagnostics':diagnostics,'messages_deleted':0,'generated_summary':False}
        # Preserve the exact input and decision before mutating the agent's active history.
        write_json(self.run_dir/snapshot,before);write_json(self.run_dir/(stem+'.json'),report)
        for i in indices:history[i]={**history[i],'content':MARKER}

    def before_forced_exposure(self,agent,history):
        indices=[i for i,m in enumerate(history) if m.get('role')=='tool'
                 and m.get('tool_call_id','').startswith('host-log-') and m.get('content')!=MARKER]
        self.archive_mask(agent,history,indices,'superseded_forced_history_before_new_payload')

    def candidates(self,agent,history):
        # A missing boundary protects everything, rather than guessing which evidence is old.
        boundary=self.starts.get(agent,0)
        return [i for i,m in enumerate(history[:boundary]) if m.get('role')=='tool'
                and m.get('tool_name') in (BROWSER_TOOLS|NOTEBOOK_TOOL_NAMES) and m.get('content')!=MARKER
                and not m.get('tool_call_id','').startswith('host-log-')]

    def fit(self,agent,history,deadline,num_predict,force=False):
        target=min(int(self.policy.context_length*0.75),self.policy.context_length-num_predict-1024)
        if target<1:raise ContextExhausted({'reason':'requested output leaves no bounded input capacity','prompt_target':target})
        while True:
            remaining=deadline-time.monotonic()
            if remaining<=0:raise TimeoutError('Bounded context fitting exhausted request time')
            try:diagnostics=self.client.count_context(history,timeout=min(15,remaining))
            except ContextExhausted as error:diagnostics=error.diagnostics
            count=diagnostics.get('prompt_tokens')
            if type(count) is not int or count<0:raise ValueError('Bounded context requires native prompt token count')
            if count<=target and not force:return diagnostics
            eligible=self.candidates(agent,history)
            if not eligible:
                failure={'reason':'protected_input_exceeds_bounded_context_target','prompt_tokens':count,
                         'prompt_target':target,'context_length':self.policy.context_length,
                         'protected_question_start':self.starts.get(agent),'generation_requested':num_predict,
                         'native_diagnostics':diagnostics,'no_summary_or_reset_performed':True}
                self.serial+=1;stem=f'bounded-context-{self.serial:04d}-{agent}-failure'
                write_json(self.run_dir/(stem+'-history.json'),history)
                write_json(self.run_dir/(stem+'.json'),failure)
                raise ContextExhausted(failure)
            self.archive_mask(agent,history,[eligible[0]],'native_input_pressure',diagnostics)
            force=False

    def request(self,agent,history,deadline,kwargs):
        remaining=deadline-time.monotonic()
        if remaining<=0:raise TimeoutError('Bounded context fitting exhausted request time')
        self.serial+=1
        write_json(self.run_dir/f'bounded-context-{self.serial:04d}-{agent}-request.json',
                   {'agent':agent,'history':history,'num_predict':kwargs['num_predict'],
                    'final_only':kwargs.get('final_only',False),'format_schema':kwargs.get('format_schema'),
                    'purpose':'exact history passed to transport after bounded fitting; transport still enforces its own native preflight'})
        return self.client(agent,history,remaining,**kwargs)

    def __call__(self,agent,history,timeout,**kwargs):
        deadline=time.monotonic()+timeout
        self.fit(agent,history,deadline,kwargs['num_predict'])
        try:return self.request(agent,history,deadline,kwargs)
        except ContextExhausted:
            # The exact request template may differ from count_context (e.g. JSON notes).
            # ContextExhausted is pre-generation; one additional reduction/recheck is safe.
            self.fit(agent,history,deadline,kwargs['num_predict'],force=True)
            return self.request(agent,history,deadline,kwargs)
