"""Bounded host delivery of one previously published true-peer notebook entry."""
import hashlib
import json
from pathlib import Path
import time

POLICY='latest-peer-body-2000-v1'


def digest(text):return hashlib.sha256(text.encode()).hexdigest()


def candidates(db,recipient,q,stage):
    return db.execute('SELECT e.slug,e.author,e.question_round,e.stage,r.id,p.title,p.body,e.created_at FROM entry_provenance e JOIN pages p USING(slug) JOIN revisions r USING(slug) WHERE e.author!=? AND (e.question_round<? OR (e.question_round=? AND e.stage<?)) ORDER BY e.question_round DESC,e.stage DESC,r.id DESC,e.slug DESC',(recipient,q,q,stage)).fetchall()


def packet(row,labels):
    slug,author,q,stage,revision,title,body,stamp=row
    prefix=body[:2000]
    metadata={'url':'https://wiki.test/page/'+slug,'title':title,'author':labels[author],'revision':f'r-{revision}',
              'created_at':stamp,'body_characters':len(body),'delivered_characters':len(prefix),'truncated':len(body)>len(prefix)}
    message={'role':'user','content':'Research document\n'+json.dumps(metadata,ensure_ascii=False,sort_keys=True)+'\n\n'+prefix}
    return message,metadata


def deliver(db,client,history,settings,recipient,q,stage,path):
    rows=candidates(db,recipient,q,stage)
    ledger={'policy':POLICY,'boundary_id':f'{q}:{stage}:{recipient}','question_round':q,'stage':stage,'recipient':recipient,
            'candidate_order':[{'slug':r[0],'author':r[1],'round':r[2],'stage':r[3],'revision':r[4]} for r in rows],
            'selection':'latest prior publication; no relevance or correctness filter','candidate_available':bool(rows),'delivered':False,'generation_calls':0}
    output=Path(path)/'peer-note-delivery.json'
    if not rows:
        ledger.update(packet=None,added_packets=0,tokenizer_requests=0,input_token_cost={'marginal_input_tokens':0,'method':'no packet added; no tokenizer request'})
        output.write_text(json.dumps(ledger,ensure_ascii=False,indent=2)+'\n');return ledger
    message,metadata=packet(rows[0],settings['visible_labels'])
    ledger.update(packet=message,packet_sha256=digest(message['content']),document=metadata,sender=rows[0][1],
                  full_body_sha256=digest(rows[0][6]),prefix_sha256=digest(rows[0][6][:2000]),added_packets=0,tokenizer_requests=0)
    started=time.monotonic()
    try:
        ledger['tokenizer_requests']+=1
        before=client.count_context(history,timeout=15)
        ledger['tokenizer_requests']+=1
        after=client.count_context(history+[message],timeout=15)
        for value in (before,after):
            if type(value.get('prompt_tokens')) is not int or value['prompt_tokens']<0:raise ValueError('Peer exposure requires native context token counts')
        ledger['native_context']={'before':before,'after':after,'marginal_input_tokens':after['prompt_tokens']-before['prompt_tokens'],
            'scope':'whole packet including wrapper in pre-phase history and tool schema; excludes upcoming phase prompt; not output tokens'}
        target=min(int(settings['policy']['context_length']*.75),settings['policy']['context_length']-settings['policy']['answer_generated_tokens' if stage==2 else 'reflection_generated_tokens']-1024)
        if after['prompt_tokens']>target:raise ValueError('Peer exposure exceeds native bounded context capacity')
        history.append(message)
        ledger.update(status='delivered',delivered=True,added_packets=1)
    except BaseException as error:
        ledger.update(status='failed',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        ledger['measurement_seconds']=time.monotonic()-started
        output.write_text(json.dumps(ledger,ensure_ascii=False,indent=2)+'\n')
    return ledger


def validate_ledger(db,ledger,settings,completed_rounds):
    expected={f'{q}:{stage}:{agent}' for q in range(1,completed_rounds+1) for stage in (2,3) for agent in settings['notebooks']}
    if not isinstance(ledger,list) or len(ledger)!=len(expected) or {r.get('boundary_id') for r in ledger}!=expected:raise ValueError('Invalid peer delivery ledger boundaries')
    for record in ledger:
        q,stage,agent=record['question_round'],record['stage'],record['recipient']
        if record['boundary_id']!=f'{q}:{stage}:{agent}' or record.get('policy')!=POLICY:raise ValueError('Invalid peer delivery identity')
        rows=candidates(db,agent,q,stage)
        ordering=[{'slug':r[0],'author':r[1],'round':r[2],'stage':r[3],'revision':r[4]} for r in rows]
        if record.get('candidate_order')!=ordering:raise ValueError('Invalid peer delivery candidate order')
        if bool(rows)!=record['delivered']:raise ValueError('Invalid peer delivery selection')
        if not rows:
            if record['packet'] is not None or record['tokenizer_requests']!=0 or record.get('added_packets')!=0 or record.get('input_token_cost',{}).get('marginal_input_tokens')!=0:raise ValueError('Invalid empty peer delivery')
            continue
        counts=record.get('native_context',{})
        for name in ('before','after'):
            if type(counts.get(name,{}).get('prompt_tokens')) is not int or counts[name]['prompt_tokens']<0:raise ValueError('Invalid peer delivery native count')
        if counts.get('marginal_input_tokens')!=counts['after']['prompt_tokens']-counts['before']['prompt_tokens'] or record.get('tokenizer_requests')!=2:raise ValueError('Invalid peer delivery token accounting')
        message,metadata=packet(rows[0],settings['visible_labels'])
        if (record['packet']!=message or record['document']!=metadata or record['sender']!=rows[0][1]
            or record['packet_sha256']!=digest(message['content']) or record['full_body_sha256']!=digest(rows[0][6])
            or record['prefix_sha256']!=digest(rows[0][6][:2000]) or record.get('status')!='delivered'):
            raise ValueError('Peer delivery payload/provenance mismatch')
