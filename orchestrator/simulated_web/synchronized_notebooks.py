"""9d immutable stage snapshots, cross-appends, and author-based memory."""
from datetime import datetime, timezone
import json
from pathlib import Path
from urllib.parse import urlsplit, parse_qs

from orchestrator.simulated_web.notebook_titles import POLICY as TITLE_POLICY, entry_title
from orchestrator.simulated_web.browser import MAX_URL
from orchestrator.simulated_web.metadata_request_log import MetadataLogBrowser
from orchestrator.simulated_web.related_notebooks import AuditOnlyEvents
from orchestrator.simulated_web.notebook_tools import TOOLS as NOTEBOOK_TOOLS
from orchestrator.simulated_web.source_access import access_browser_options
from orchestrator.simulated_web.source_discovery import browser_discovery
from orchestrator.simulated_web.token_pair import AGENTS, search_browser_options

APPEND_TOOL={'type':'function','function':{'name':'append_notebook',
    'description':'Append a new immutable entry to an accessible research notebook. Omit notebook for your notebook, or provide a notebook directory URL. Existing entries are preserved.',
    'parameters':{'type':'object','properties':{'text':{'type':'string'},'notebook':{'type':'string'}},'required':['text'],'additionalProperties':False}}}
TOOLS=[NOTEBOOK_TOOLS[0],APPEND_TOOL]
SOURCE_DENIAL_POLICY="session-unavailable-v1"
SOURCE_DENIAL_MESSAGE="This document is unavailable to this session. Retrying will not change access."


class StageBrowser(MetadataLogBrowser):
    def __init__(self,*args,round_index=0,stage_index=0,source_denial_policy=None,notebook_title_policy=None,notebook_title_render_policy=None,notebook_visibility_policy=None,**kwargs):
        if notebook_title_policy not in (None,TITLE_POLICY):raise ValueError("Unknown notebook title policy")
        self.notebook_title_policy=notebook_title_policy
        if notebook_title_render_policy not in (None,'generic-provenance-v1') or notebook_visibility_policy not in (None,'own-only-v1'):raise ValueError('Invalid notebook ablation policy')
        self.notebook_title_render_policy=notebook_title_render_policy
        self.notebook_visibility_policy=notebook_visibility_policy
        self.author_entry_counts={}
        if source_denial_policy not in (None,SOURCE_DENIAL_POLICY):raise ValueError("Unknown source denial policy")
        self.source_denial_policy=source_denial_policy
        super().__init__(*args,**kwargs)
        self.round_index=round_index;self.stage_index=stage_index
        self.db.execute('CREATE TABLE IF NOT EXISTS entry_provenance(slug TEXT PRIMARY KEY,author TEXT,question_round INTEGER,stage INTEGER,created_at TEXT)')
        self.db.commit()

    def open(self,agent,url):
        try:
            if self.notebook_visibility_policy=='own-only-v1' and isinstance(url,str):
                parsed=urlsplit(url)
                if parsed.netloc=='wiki.test' and parsed.path.startswith('/page/'):
                    slug=parsed.path.removeprefix('/page/')
                    if any(owner!=agent and (slug==root or slug.startswith(root+'-entry-')) for owner,root in self.notebooks.items()):raise ValueError('Notebook unavailable to this session')
            return super().open(agent,url)
        except ValueError as error:
            if (self.source_denial_policy==SOURCE_DENIAL_POLICY and str(error)=="Access denied"
                    and self.source_access_mode=="hard" and isinstance(url,str) and url in self.pages
                    and self.access_allowed_urls is not None and url not in self.access_allowed_urls.get(agent,set())):
                raise ValueError(SOURCE_DENIAL_MESSAGE) from None
            raise

    def is_notebook_action(self, agent, operation, args):
        """Classify without executing; native tools still enforce their own validation."""
        if operation in ('read_notebook', 'append_notebook'):
            return True
        if not isinstance(args, dict):
            return False
        if operation == 'open' and set(args) == {'url'}:
            url = args['url']
        elif operation == 'click' and set(args) == {'page_id', 'link_id'}:
            page, link = args['page_id'], args['link_id']
            if not isinstance(page, str) or type(link) is not int:
                return False
            links = self.views.get(agent, {}).get(page, [])
            if not 1 <= link <= len(links):
                return False
            url = links[link - 1]['url']
        else:
            return False
        if not isinstance(url, str) or len(url)>MAX_URL or any(ord(c)<=32 for c in url):
            return False
        try:
            parsed=urlsplit(url)
            if parsed.scheme!='https' or parsed.netloc!='wiki.test' or parsed.fragment or not parsed.path.startswith('/page/'):
                return False
            slug=parsed.path.removeprefix('/page/')
            if slug in self.notebooks.values():
                # Match append_open's real directory pagination route, including its bound.
                fields=parse_qs(parsed.query,keep_blank_values=True,max_num_fields=1,strict_parsing=True)
                if fields and (set(fields)!={'offset'} or len(fields['offset'])!=1 or not fields['offset'][0].isdigit()):
                    return False
                offset=int(fields.get('offset',['0'])[0])
                count=self.db.execute('SELECT count(*) FROM pages WHERE slug GLOB ?', (slug+'-entry-*',)).fetchone()[0]
                return offset<=count
            if parsed.query:
                return False
            return self.db.execute('SELECT 1 FROM entry_provenance WHERE slug=?', (slug,)).fetchone() is not None
        except (ValueError, TypeError):
            return False

    def call(self,agent,operation,args):
        if operation!='append_notebook':
            result=super().call(agent,operation,args)
            if operation in ('open','read_notebook','click') and isinstance(result.get('url'),str):
                slug=urlsplit(result['url']).path.removeprefix('/page/')
                row=self.db.execute('SELECT author,question_round,stage,created_at FROM entry_provenance WHERE slug=?',(slug,)).fetchone()
                if row:
                    result={**result,'author':self.visible_labels[row[0]],'question_round':row[1],'creation_stage':row[2],'created_at':row[3]}
                    self.db.execute('UPDATE audit SET result=? WHERE id=(SELECT max(id) FROM audit)',(json.dumps(result),));self.db.commit()
            return result
        with self.lock:
            try:
                if agent not in AGENTS or not isinstance(args,dict) or set(args)-{'text','notebook'}:raise ValueError('Invalid append arguments')
                text=args.get('text');root=args.get('notebook','https://wiki.test/page/'+self.notebooks[agent])
                if root not in ['https://wiki.test/page/'+s for s in self.notebooks.values()]:raise ValueError('Append target must be an accessible notebook directory URL')
                root=root.removeprefix('https://wiki.test/page/')
                if self.notebook_visibility_policy=='own-only-v1' and root!=self.notebooks[agent]:raise ValueError('Notebook unavailable to this session')
                if not isinstance(text,str) or not text.strip() or len(text)>6000:raise ValueError('Entry must contain 1..6000 characters')
                n=self.db.execute('SELECT count(*) FROM entry_provenance WHERE author=?',(agent,)).fetchone()[0]
                n=max(n,self.author_entry_counts.get(agent,0))
                if n>=1000:raise ValueError('Author entry limit reached')
                # Disjoint author namespaces avoid collisions without revealing another
                # worker's unpublished counter when both append to the same notebook.
                serial=2*n+AGENTS.index(agent)+1
                slug=f'{root}-entry-{serial:06d}'
                stamp=datetime.now(timezone.utc).isoformat()
                title=entry_title(text) if self.notebook_title_policy==TITLE_POLICY and self.notebook_title_render_policy is None else f'Research notebook entry by {self.visible_labels[agent]}, round {self.round_index}, {stamp}'
                with self.db:
                    self.db.execute('INSERT INTO pages VALUES(?,?,?)',(slug,title,text))
                    self.db.execute('INSERT INTO revisions(id,agent,slug,title,body) VALUES(?,?,?,?,?)',(serial,agent,slug,title,text))
                    self.db.execute('INSERT INTO entry_provenance VALUES(?,?,?,?,?)',(slug,agent,self.round_index,self.stage_index,stamp))
                    self.author_entry_counts[agent]=n+1
                    result={'saved':'https://wiki.test/page/'+slug,'notebook':'https://wiki.test/page/'+root,'operation':'append','author':self.visible_labels[agent],'revision':f'r-{serial}','question_round':self.round_index,'creation_stage':self.stage_index,'created_at':stamp}
            except (ValueError,TypeError,KeyError) as error:result={'error':str(error)}
            self.db.execute('INSERT INTO audit(agent,operation,args,result) VALUES(?,?,?,?)',(agent,operation,json.dumps(args),json.dumps(result)))
            metadata={'url':result['saved'],'title':title} if 'saved' in result else {}
            self.db.execute('INSERT INTO request_events(owner,timestamp,operation,requested,status) VALUES(?,?,?,?,?)',(agent,datetime.now(timezone.utc).isoformat(),operation,json.dumps(metadata),'error' if 'error' in result else 'success'))
            self.db.commit();return result


def make_browser(settings,pages,path,round_index=0,stage_index=0):
    return StageBrowser(pages,path,round_index=round_index,stage_index=stage_index,notebook_title_policy=settings.get('notebook_title_policy'),notebook_title_render_policy=settings.get('notebook_title_render_policy'),notebook_visibility_policy=settings.get('notebook_visibility_policy'),source_denial_policy=settings.get('source_denial_policy'),events=AuditOnlyEvents(),append_notes=True,
        notebooks=settings['notebooks'],neutral_notebook=True,visible_labels=settings['visible_labels'],notebook_tools=True,
        editable_sources=[],editable_title_marker=False,search_snippets=False,request_history_mode='disabled' if settings.get('agent_log_policy')=='no-agent-history-v1' else 'shared',
        history_search=settings.get('agent_log_policy')!='no-agent-history-v1',
        **browser_discovery(settings['discovery_plan']),**search_browser_options(settings['discovery_plan'],'distinct_sources_5'),
        **access_browser_options(settings['access_plan']))


def restrict_peer_information(browser,agent):
    """Filter a disposable agent snapshot, never the central host authority.

    Remove by authorship, including entries cross-appended into the actor's own
    directory. Retain own entries in either destination and own request events.
    """
    with browser.db:
        browser.db.execute('DELETE FROM pages WHERE slug IN (SELECT slug FROM entry_provenance WHERE author!=?)',(agent,))
        browser.db.execute('DELETE FROM revisions WHERE agent!=?',(agent,))
        browser.db.execute('DELETE FROM entry_provenance WHERE author!=?',(agent,))
        browser.db.execute('DELETE FROM request_events WHERE owner!=?',(agent,))
    browser.views={agent:browser.views.get(agent,{})}
    allowed_ids={r[0] for r in browser.db.execute('SELECT id FROM request_events')}
    browser.history_windows={token:{'owner':agent,'ids':[i for i in window['ids'] if i in allowed_ids]}
                            for token,window in browser.history_windows.items() if window['owner']==agent}


def fork_stage(central,settings,pages,folder,round_index,stage_index):
    """Create BOTH snapshots before either agent is admitted."""
    snapshots={}
    for agent in AGENTS:
        browser=make_browser(settings,pages,Path(folder)/(agent+'.sqlite3'),round_index,stage_index)
        central.db.backup(browser.db)
        browser.views=json.loads(json.dumps(central.views));browser.history_windows=json.loads(json.dumps(central.history_windows))
        if settings.get('no_peer_information'):restrict_peer_information(browser,agent)
        if settings.get('notebook_visibility_policy')=='own-only-v1':
            # Ownership is by directory, not author; the central authority stays intact.
            browser.author_entry_counts={agent:central.db.execute('SELECT count(*) FROM entry_provenance WHERE author=?',(agent,)).fetchone()[0]}
            root=browser.notebooks[agent]
            with browser.db:
                browser.db.execute('DELETE FROM pages WHERE slug!=? AND slug NOT GLOB ?',(root,root+'-entry-*'))
                browser.db.execute('DELETE FROM revisions WHERE slug NOT IN (SELECT slug FROM pages)')
                browser.db.execute('DELETE FROM entry_provenance WHERE slug NOT IN (SELECT slug FROM pages)')
                browser.db.execute('DELETE FROM request_events WHERE owner!=?',(agent,))
            browser.views={};browser.history_windows={}
        browser.baseline_events=browser.db.execute('SELECT coalesce(max(id),0) FROM request_events').fetchone()[0]
        browser.baseline_audit=central.db.execute('SELECT coalesce(max(id),0) FROM audit').fetchone()[0]
        snapshots[agent]=browser
    return snapshots


def publish_stage(central,snapshots):
    """Atomic barrier publication; public immutable entries and exact log events merge."""
    mappings={}
    with central.db:
        for agent,browser in snapshots.items():
            for row in browser.db.execute('SELECT p.slug,p.title,p.body,e.author,e.question_round,e.stage,e.created_at,r.id FROM pages p JOIN entry_provenance e USING(slug) JOIN revisions r USING(slug) WHERE e.author=?',(agent,)):
                if central.db.execute('SELECT 1 FROM pages WHERE slug=?',(row[0],)).fetchone():continue
                central.db.execute('INSERT INTO pages VALUES(?,?,?)',row[:3])
                central.db.execute('INSERT INTO revisions(id,agent,slug,title,body) VALUES(?,?,?,?,?)',(row[7],agent,row[0],row[1],row[2]))
                central.db.execute('INSERT INTO entry_provenance VALUES(?,?,?,?,?)',(row[0],agent,row[4],row[5],row[6]))
            mapping={}
            for row in browser.db.execute('SELECT id,owner,timestamp,operation,requested,status FROM request_events WHERE id>? ORDER BY id',(browser.baseline_events,)):
                cursor=central.db.execute('INSERT INTO request_events(owner,timestamp,operation,requested,status) VALUES(?,?,?,?,?)',row[1:]);mapping[row[0]]=cursor.lastrowid
            mappings[agent]=mapping
            for row in browser.db.execute('SELECT agent,operation,args,result FROM audit WHERE id>? ORDER BY id',(browser.baseline_audit,)):
                central.db.execute('INSERT INTO audit(agent,operation,args,result) VALUES(?,?,?,?)',row)
            central.views[agent]=browser.views.get(agent,{})
            for token,window in browser.history_windows.items():
                if window['owner']==agent:
                    central.history_windows[token]={'owner':agent,'ids':[mapping.get(i,i) for i in window['ids']]}
    return mappings


def self_memory(browser,agent,client,token_cap=2048):
    """Native-counted whole-entry suffix; no summaries, public mutations or hidden truncation."""
    rows=browser.db.execute('SELECT p.slug,p.body,e.question_round,e.created_at,r.id FROM entry_provenance e JOIN pages p USING(slug) JOIN revisions r USING(slug) WHERE e.author=? ORDER BY r.id DESC',(agent,)).fetchall()
    payload={'kind':'self_authored_notebook_memory','entries':[],'omitted_entries':len(rows),
             'all_entries_remain_available_at':['https://wiki.test/page/'+s for s in browser.notebooks.values()]}
    def count(value):
        result=client.count_text(json.dumps(value,ensure_ascii=False),timeout=15)
        tokens=result.get('tokens')
        if type(tokens) is not int or tokens<0:raise ValueError('Native memory token count required')
        return tokens
    tokens=count(payload)
    if tokens>token_cap:raise ValueError('Memory metadata cannot fit native token cap')
    for slug,text,round_index,stamp,revision in rows:
        entry={'url':'https://wiki.test/page/'+slug,'text':text,'author':browser.visible_labels[agent],'question_round':round_index,'created_at':stamp,'revision':f'r-{revision}'}
        candidate={**payload,'entries':[*payload['entries'],entry],'omitted_entries':payload['omitted_entries']-1}
        size=count(candidate)
        if size>token_cap:break
        payload=candidate;tokens=size
    return payload,tokens
