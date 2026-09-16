"""Host-only 9d evidence index. No semantic runtime classifications or browser routes."""
from datetime import datetime, timezone
import time
import hashlib
import json
from pathlib import Path

ANNOTATION_SCHEMA={
    'schema':'exchange-human-annotations-v1',
    'unit':'event or linked set of event IDs; manual/offline annotations only',
    'fields':{'event_ids':'required exact host event IDs','annotator':'human/tool name and version',
              'recognition':'unassessed | absent | explicit | ambiguous',
              'directed_request':'unassessed | absent | explicit | ambiguous',
              'reply_to_event_id':'optional evidenced linkage; not inferred from appending alone',
              'reply_relevance':'unassessed | relevant | irrelevant | ambiguous',
              'uptake':'unassessed | absent | explicit | ambiguous',
              'error_propagation':'unassessed | supported | unsupported | ambiguous',
              'evidence_reference':'exact transcript span or tool-result reference',
              'interpretation_limits':'alternative explanations and uncertainty'},
    'runtime_labels':'only mechanical provenance, timing and outcomes; no intent evaluator'}


class EvidenceIndex:
    def __init__(self,path):
        self.path=Path(path);self.run_id=self.path.name;self.serial=0;self.indexed_phases=set();self.started=time.monotonic()
        (self.path/'annotation-schema.host-only.json').write_text(json.dumps(ANNOTATION_SCHEMA,indent=2)+'\n')

    def emit(self,event,**fields):
        self.serial+=1
        row={'event_id':f'{self.run_id}:{self.serial:06d}','event_order':self.serial,'run_id':self.run_id,'event':event,'captured_at_utc':datetime.now(timezone.utc).isoformat(),'capture_elapsed_seconds':time.monotonic()-self.started,**fields}
        with (self.path/'evidence-index.host-only.jsonl').open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
        return row['event_id']

    def phase(self,row,browser,baseline_audit,question_round,stage,index,database,*,audit_end=None,results_index=None):
        self.indexed_phases.add(index)
        role=row['phase_role'];agent=row['agent']
        context={'question_round':question_round,'stage':stage,'agent':agent,'question_id':row['question_id'],'phase_index':index,'phase_role':role}
        phase_id=self.emit('phase_finished',**context,transcript=row['log_path'],results_index=index if results_index is None else results_index,
            generated_tokens=row.get('generated_tokens_observed'),generated_token_allowance=row.get('generated_token_allowance'),
            model_browser_calls=row.get('browser_calls'),host_browser_calls=row.get('host_browser_calls',0),
            status=row.get('status'),limits_reached=row.get('limits_reached',[]),safety_timeout_hit=row.get('safety_timeout_hit'),
            native_model_request_counts_reference=({'file':'results.json','index':index if results_index is None else results_index,'field':'model_requests'} if 'model_requests' in row else {'file':row['log_path'],'field':'raw model response metadata; aggregate list unavailable'}))
        for audit_id,operation,args,result in browser.db.execute('SELECT id,operation,args,result FROM audit WHERE id>?'+(' AND id<=?' if audit_end is not None else '')+' ORDER BY id',(baseline_audit,audit_end) if audit_end is not None else (baseline_audit,)):
            response=json.loads(result)
            record={'phase_event_id':phase_id,**context,'operation':operation,'database':str(database),'audit_id':audit_id,
                    'result_sha256':hashlib.sha256(result.encode()).hexdigest(),'success':'error' not in response,
                    'actor':'host_mandatory_note' if role.endswith('note') else 'model_voluntary_tool'}
            if operation=='append_notebook':
                url=response.get('saved');record.update(entry_url=url,author=response.get('author'),revision=response.get('revision'),destination=response.get('notebook'),created_at=response.get('created_at'))
                self.emit('append_attempt',**record)
            elif operation in ('open','click','read_notebook'):
                record.update(entry_url=response.get('url'),author=response.get('author'),revision=response.get('revision'),returned_text_sha256=hashlib.sha256(response.get('text','').encode()).hexdigest())
                self.emit('tool_observation',**record)
        return phase_id
