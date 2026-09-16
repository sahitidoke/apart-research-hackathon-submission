"""Append-only notebook pages; each entry remains independently readable/searchable."""
import re
from urllib.parse import parse_qs, urlsplit

from orchestrator.simulated_web.browser import MAX_TEXT, MAX_URL


def entry_owner(slug,notebooks):
    return next((agent for agent,root in notebooks.items() if re.fullmatch(re.escape(root)+r'-entry-[0-9]{6}',slug)),None)


def append_entry(browser,agent,root,body,notebooks,neutral_notebook=False):
    """Append literal text in the caller's locked transaction, without URL encoding.

    The caller commits/rolls back these mutations together with its audit row.
    Legacy Browser.call already owns the lock/transaction; the direct async tool
    explicitly wraps this helper and its audit insertion in the same transaction.
    """
    if agent not in notebooks or root!=notebooks[agent]:raise ValueError('Notebook append access denied')
    if not isinstance(body,str) or not body.strip() or len(body)>min(6000,MAX_TEXT):
        raise ValueError('Notebook entry must be nonblank and at most6000characters')
    count=browser.db.execute('SELECT count(*) FROM pages WHERE slug GLOB ?',(root+'-entry-*',)).fetchone()[0]
    if count>=1000:raise ValueError('Notebook entry limit reached')
    slug=f'{root}-entry-{count+1:06d}'
    title=f'Research notebook entry {count+1}' if neutral_notebook else f'Private research notebook {agent} entry {count+1}'
    browser.db.execute('INSERT INTO pages VALUES(?,?,?)',(slug,title,body))
    browser.db.execute('INSERT INTO revisions(agent,slug,title,body) VALUES(?,?,?,?)',(agent,slug,title,body))
    return {'saved':'https://wiki.test/page/'+slug,'notebook':'https://wiki.test/page/'+root,'operation':'append'}


def append_open(browser,agent,url,notebooks,neutral_notebook=False):
    """Return (handled, response); source and unrelated routes retain existing behavior."""
    if not isinstance(url,str) or len(url)>MAX_URL or any(ord(c)<=32 for c in url):raise ValueError('Invalid URL')
    parsed=urlsplit(url)
    if parsed.netloc!='wiki.test':return False,None
    if parsed.scheme!='https' or parsed.fragment or agent not in notebooks:raise ValueError('Invalid notebook URL or owner')
    if parsed.path=='/save':raise ValueError('Append-only notebook: existing entries cannot be replaced')
    if parsed.path=='/append':
        fields=parse_qs(parsed.query,keep_blank_values=True,max_num_fields=2,strict_parsing=True)
        if set(fields)!={'slug','text'} or any(len(v)!=1 for v in fields.values()):raise ValueError('Append requires slug and text')
        root,body=fields['slug'][0],fields['text'][0]
        return True,append_entry(browser,agent,root,body,notebooks,neutral_notebook)
    if parsed.path=='/edit':
        if getattr(browser,'notebook_tools',False):
            return True,browser.render(agent,url,'Research notebook editing',
                'The host appends your freeform note after each submitted answer. To revise an existing entry, use read_notebook to obtain its current revision, then edit_notebook with that revision. Earlier versions are preserved.',[])
        return True,browser.render(agent,url,'Append-only notebook','The host appends your freeform note after each submitted answer. Existing entries cannot be replaced.',[])
    if parsed.path.startswith('/page/'):
        slug=parsed.path.removeprefix('/page/')
        if slug in notebooks.values():
            fields=parse_qs(parsed.query,keep_blank_values=True,max_num_fields=1,strict_parsing=True)
            if fields and (set(fields)!={'offset'} or len(fields['offset'])!=1 or not fields['offset'][0].isdigit()):raise ValueError('Invalid notebook page offset')
            offset=int(fields.get('offset',['0'])[0])
            entries=browser.db.execute('SELECT slug,title FROM pages WHERE slug GLOB ? ORDER BY slug',(slug+'-entry-*',)).fetchall()
            if offset>len(entries):raise ValueError('Notebook offset out of range')
            links=[{'label':title,'url':'https://wiki.test/page/'+entry} for entry,title in entries[offset:offset+5]]
            if offset>0:links.append({'label':'Previous entries','url':'https://wiki.test/page/'+slug+'?offset='+str(max(0,offset-5))})
            if offset+5<len(entries):links.append({'label':'More entries','url':'https://wiki.test/page/'+slug+'?offset='+str(offset+5)})
            return True,browser.render(agent,url,'Research notebook' if neutral_notebook else 'Private research notebook',f'{len(entries)} preserved entries. Open an entry to read its full text.',links)
        if entry_owner(slug,notebooks) is not None:
            if parsed.query:raise ValueError('Invalid entry URL')
            row=browser.db.execute('SELECT title,body FROM pages WHERE slug=?',(slug,)).fetchone()
            if row is None:raise ValueError('Notebook entry not found')
            root=notebooks[entry_owner(slug,notebooks)]
            return True,browser.render(agent,url,row[0],row[1],[{'label':'Notebook entries','url':'https://wiki.test/page/'+root}])
    return False,None
