"""Explicit local notebook read/edit tools with immutable revisions and conflicts."""
from datetime import datetime,timezone
import json
import re
from urllib.parse import urlsplit

from orchestrator.simulated_web.append_notebooks import entry_owner

TOOLS = [{"type":"function","function":{"name":name,"description":description,
    "parameters":{"type":"object","properties":properties,"required":list(properties),"additionalProperties":False}}}
    for name,description,properties in [
    ('read_notebook','Read a research notebook entry, including its author and revision. Use revision "" for the current version; a returned previous_revision reads the prior version.',
     {'url':{'type':'string'},'revision':{'type':'string'}}),
    ('edit_notebook','Revise an existing research notebook entry. Read it first and supply its current revision to detect conflicts. Previous versions remain preserved. This tool cannot edit source documents.',
     {'url':{'type':'string'},'expected_revision':{'type':'string'},'text':{'type':'string'}})]]
NAMES=frozenset(tool['function']['name'] for tool in TOOLS)


def revision_id(value):
    if not isinstance(value,str) or re.fullmatch('r-[1-9][0-9]*',value) is None:raise ValueError('Invalid notebook revision')
    return int(value[2:])


def notebook_action(browser,agent,operation,args):
    if not isinstance(args,dict) or set(args)!=({'url','revision'} if operation=='read_notebook' else {'url','expected_revision','text'}):
        raise ValueError('Incorrect notebook tool arguments')
    url=args['url']
    if not isinstance(url,str) or len(url)>512:raise ValueError('Invalid notebook URL')
    parsed=urlsplit(url)
    if parsed.scheme!='https' or parsed.netloc!='wiki.test' or parsed.query or parsed.fragment or not parsed.path.startswith('/page/'):
        raise ValueError('Notebook tools accept only local notebook page URLs')
    slug=parsed.path.removeprefix('/page/')
    owner=entry_owner(slug,browser.notebooks)
    if slug in browser.notebooks.values() and operation=='read_notebook':
        if args['revision']!='':raise ValueError('Notebook directory has no revision')
        return browser.open(agent,url)
    if owner is None:raise ValueError('Notebook tools cannot read or edit source documents or unknown entries')
    latest=browser.db.execute('SELECT id,agent,title,body FROM revisions WHERE slug=? ORDER BY id DESC LIMIT 1',(slug,)).fetchone()
    if latest is None:raise ValueError('Notebook entry not found')
    if operation=='read_notebook':
        selected=latest if args['revision']=='' else browser.db.execute('SELECT id,agent,title,body FROM revisions WHERE slug=? AND id=?',(slug,revision_id(args['revision']))).fetchone()
        if selected is None:raise ValueError('Revision does not belong to this notebook entry')
        previous=browser.db.execute('SELECT max(id) FROM revisions WHERE slug=? AND id<?',(slug,selected[0])).fetchone()[0]
        return {'url':url,'title':selected[2],'text':selected[3],'author':browser.visible_labels[selected[1]],
                'revision':f'r-{selected[0]}','current_revision':f'r-{latest[0]}',
                'previous_revision':f'r-{previous}' if previous is not None else ''}
    expected=revision_id(args['expected_revision'])
    if expected!=latest[0]:return {'error':'Notebook revision conflict; read the current entry before editing','current_revision':f'r-{latest[0]}'}
    text=args['text']
    if not isinstance(text,str) or not text.strip() or len(text)>6000:raise ValueError('Edited entry must be nonblank and at most6000characters')
    # Audit revisions preserve the original append and every edit, including author identity.
    browser.db.execute('UPDATE pages SET body=? WHERE slug=?',(text,slug))
    cursor=browser.db.execute('INSERT INTO revisions(agent,slug,title,body) VALUES(?,?,?,?)',(agent,slug,latest[2],text))
    return {'updated':url,'revision':f'r-{cursor.lastrowid}','author':browser.visible_labels[agent]}


def call_notebook_tool(browser,agent,operation,args):
    with browser.lock:
        try:
            if agent not in browser.visible_labels:raise ValueError('Unknown notebook actor')
            result=notebook_action(browser,agent,operation,args)
        except (ValueError,TypeError,KeyError) as error:result={'error':str(error)}
        requested=json.dumps(args,ensure_ascii=False)
        # Genuine local tool calls are public; logical actor stays host-only in the owner column.
        browser.db.execute('INSERT INTO request_events(owner,timestamp,operation,requested,status) VALUES(?,?,?,?,?)',
                           (agent,datetime.now(timezone.utc).isoformat(),operation,requested[:31000],'error' if 'error' in result else 'ok'))
        browser.db.execute('INSERT INTO audit(agent,operation,args,result) VALUES(?,?,?,?)',
                           (agent,operation,requested[:31000],json.dumps(result)))
        browser.db.commit()
        return result
