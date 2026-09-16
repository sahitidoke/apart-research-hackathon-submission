"""Finite local browser: URLs are data, never network requests or file paths."""
from collections import Counter
from datetime import datetime, timezone
from html import escape
import uuid
import hashlib
import json
import math
import re
import sqlite3
import threading
from urllib.parse import parse_qs, urlsplit, urlencode

MAX_TEXT = 8000
MAX_URL = 30000
MAX_WRITES = 100
SOURCE_EDITOR_REVISION_NAMESPACE = b"agent-swarming/source-editor-revision/v1\0"
SLUG = re.compile(r"[a-z0-9-]{1,64}\Z")
TOOLS = [{"type": "function", "function": {"name": name, "description": description,
    "parameters": {"type": "object", "properties": properties, "required": list(properties),
                   "additionalProperties": False}}} for name, description, properties in [
    ("search", "Search available pages.", {"query": {"type": "string"}}),
    ("open", "Open a URL.", {"url": {"type": "string"}}),
    ("click", "Follow a numbered link on a previously returned page.",
     {"page_id": {"type": "string"}, "link_id": {"type": "integer"}})]]


def source_editor_revision(identity):
    """Stable editor-only label, independently hashed from canonical identity."""
    digest = hashlib.sha256(SOURCE_EDITOR_REVISION_NAMESPACE + identity.encode('ascii')).hexdigest()[:12].upper()
    return 'R-' + '-'.join(digest[i:i + 4] for i in range(0, 12, 4))


def search_snippet(text, term_weights):
    """Return an unchanged source span around the strongest query-term cluster."""
    if len(text) <= 300:
        return text
    matches = [match for match in re.finditer(r"\w+", text)
               if match.group().lower() in term_weights]
    if not matches:
        return text[:300]
    starts = {0}
    for match in matches:
        # Leave room for context before a match and avoid starting inside a word.
        start = max(0, match.start() - 80)
        while start > 0 and text[start - 1].isalnum():
            start -= 1
        starts.add(start)
    def weight(start):
        terms = {match.group().lower() for match in matches
                 if start <= match.start() and match.end() <= start + 300}
        return sum(term_weights[term] for term in sorted(terms))
    start = max(sorted(starts), key=weight)
    end = min(start + 300, len(text))
    # Prefer whole words at the end, without exceeding the existing size budget.
    while end > start and end < len(text) and text[end - 1].isalnum() and text[end].isalnum():
        end -= 1
    return text[start:end] if end > start else text[start:start + 300]


class Browser:
    def __init__(self, corpus, database, editable_source=None, search_snippets=True, editable_sources=None, editable_title_marker=True, request_history_mode="disabled", discovery_hidden_urls=None, discovery_listing_urls=None, search_policy="legacy_pages_10", search_source_groups=None, access_allowed_urls=None, shared_wiki=False, source_access_mode="hard", history_search=False):
        if request_history_mode not in ("disabled", "shared", "isolated"):
            raise ValueError("Invalid request history mode")
        if search_policy not in ('legacy_pages_10', 'distinct_sources_5'):
            raise ValueError('Invalid search policy')
        if type(shared_wiki) is not bool:
            raise ValueError("shared_wiki must be boolean")
        if source_access_mode not in ("hard", "discovery_only"):
            raise ValueError("Invalid source access mode")
        if type(history_search) is not bool or (history_search and request_history_mode == "disabled"):
            raise ValueError("History search requires enabled request history")
        self.history_search = history_search
        self.source_access_mode = source_access_mode
        self.shared_wiki = shared_wiki
        self.search_policy = search_policy
        self.search_source_groups = dict(search_source_groups or {})
        if any(not isinstance(url, str) or not isinstance(identity, str) or not identity
               for url, identity in self.search_source_groups.items()):
            raise ValueError('Invalid search source groups')
        self.request_history_mode = request_history_mode
        self.history_windows = {}
        if type(search_snippets) is not bool:
            raise ValueError("search_snippets must be a boolean")
        if type(editable_title_marker) is not bool:
            raise ValueError("editable_title_marker must be boolean")
        self.editable_title_marker = editable_title_marker
        self.search_snippets = search_snippets
        self.pages = {page["url"]: page for page in corpus}
        self.access_allowed_urls = None if access_allowed_urls is None else {agent: set(urls) for agent, urls in access_allowed_urls.items()}
        if self.access_allowed_urls is not None and (not self.access_allowed_urls or any(
                not isinstance(agent, str) or not urls <= self.pages.keys() for agent, urls in self.access_allowed_urls.items())):
            raise ValueError('Invalid hard access URLs')
        if len(self.pages) != len(corpus) or len(corpus) > 10000:
            raise ValueError("Duplicate URLs or too many corpus pages")
        if not self.search_source_groups.keys() <= self.pages.keys():
            raise ValueError('Search source groups reference missing pages')
        for page in corpus:
            url = urlsplit(page["url"])
            if url.scheme != "https" or url.netloc != "docs.test" or url.query or url.fragment:
                raise ValueError("Corpus URLs must be plain https://docs.test URLs")
            if len(page["text"]) > MAX_TEXT or len(page["title"]) > 200:
                raise ValueError("Corpus page exceeds limits")
        if editable_source is not None and editable_sources is not None:
            raise ValueError('Use either legacy editable_source or editable_sources')
        selectors = [editable_source] if editable_source is not None else (editable_sources or [])
        self.editable_urls = set()
        self.source_identities = {}
        self.source_urls = {}
        for title, text_hash in selectors:
            matches = [p for p in corpus if p['title'] == title and
                       hashlib.sha256(p['text'].encode()).hexdigest() == text_hash]
            if not matches:
                raise ValueError('Editable source title/text hash not found')
            identity = hashlib.sha256((title + '\0' + matches[0]['text']).encode()).hexdigest()
            self.source_urls[identity] = sorted(p['url'] for p in matches)
            self.source_identities.update({p['url']: identity for p in matches})
            self.editable_urls.update(p['url'] for p in matches)
        self.source_editor_revisions = {identity: source_editor_revision(identity) for identity in self.source_urls}
        if len(set(self.source_editor_revisions.values())) != len(self.source_editor_revisions):
            raise ValueError('Editable source revision label collision')
        # Preserve the legacy single-source attribute for existing callers.
        self.source_identity = next(iter(self.source_urls), None)
        self.discovery_hidden_urls = {agent: set(urls) for agent, urls in (discovery_hidden_urls or {}).items()}
        self.discovery_listing_urls = set(discovery_listing_urls or [])
        if (any(url not in self.pages for urls in self.discovery_hidden_urls.values() for url in urls)
                or not self.discovery_listing_urls <= self.pages.keys()):
            raise ValueError('Unknown source discovery URL')
        self.search_tokens = {url: Counter(re.findall(r"\w+", (page["title"] + " " + page["text"]).lower()))
                              for url, page in self.pages.items()}
        self.lock = threading.RLock()
        self.db = sqlite3.connect(database, check_same_thread=False)
        self.db.executescript("""
            CREATE TABLE pages(slug TEXT PRIMARY KEY, title TEXT, body TEXT);
            CREATE TABLE revisions(id INTEGER PRIMARY KEY, agent TEXT, slug TEXT, title TEXT, body TEXT);
            CREATE TABLE source_pages(identity TEXT PRIMARY KEY, title TEXT, body TEXT);
            CREATE TABLE audit(id INTEGER PRIMARY KEY, agent TEXT, operation TEXT, args TEXT, result TEXT);
        """)
        self.db.execute("CREATE TABLE request_events(id INTEGER PRIMARY KEY, owner TEXT, timestamp TEXT, operation TEXT, requested TEXT, status TEXT)")
        for identity, urls in self.source_urls.items():
            source = self.pages[urls[0]]
            self.db.execute('INSERT INTO source_pages VALUES(?,?,?)',
                            (identity, source['title'], source['text']))
        if shared_wiki:
            self.db.executemany("INSERT INTO pages VALUES(?,?,?)",
                                [(f"notes-{i}", f"Notes {i}", "") for i in range(1, 4)])
        self.db.commit()
        self.views = {}

    def reset_views(self):
        """Host-only assignment boundary; wiki state and write budgets persist."""
        with self.lock:
            self.views.clear()

    def checkpoint(self):
        """Host-only audit/revision boundaries, never exposed as browser tools."""
        with self.lock:
            return {table: self.db.execute(f"SELECT COALESCE(MAX(id), 0) FROM {table}").fetchone()[0]
                    for table in ("audit", "revisions")}

    def close(self):
        self.db.close()

    def call(self, agent, operation, args):
        with self.lock:
            requested = None
            try:
                expected = {"search": {"query"}, "open": {"url"}, "click": {"page_id", "link_id"}}
                if operation not in expected or not isinstance(args, dict) or set(args) != expected[operation]:
                    raise ValueError("Unknown operation or incorrect arguments")
                if len(json.dumps(args)) > MAX_URL + 1000:
                    raise ValueError("Request too large")
                if operation == "search":
                    if isinstance(args["query"], str) and len(args["query"]) <= 500:
                        requested = args["query"]
                    result = self.search(args["query"], agent=agent)
                elif operation == "open":
                    if isinstance(args["url"], str) and len(args["url"]) <= MAX_URL:
                        requested = args["url"]
                    result = self.open(agent, args["url"])
                else:
                    page_id, link = args["page_id"], args["link_id"]
                    if not isinstance(page_id, str) or type(link) is not int:
                        raise ValueError("Invalid link reference")
                    links = self.views.get(agent, {}).get(page_id, [])
                    if not 1 <= link <= len(links):
                        raise ValueError("Unknown link or private page reference")
                    requested = links[link - 1]["url"]
                    result = self.open(agent, requested)
            except (ValueError, TypeError, KeyError) as error:
                result = {"error": str(error)}
            if self.request_history_mode != "disabled" and requested is not None:
                self.db.execute("INSERT INTO request_events(owner,timestamp,operation,requested,status) VALUES(?,?,?,?,?)",
                                (agent, datetime.now(timezone.utc).isoformat(), operation, requested,
                                 "error" if "error" in result else "success"))
            # Oversized invalid requests are represented by a bounded prefix in the audit.
            self.db.execute("INSERT INTO audit(agent,operation,args,result) VALUES(?,?,?,?)",
                (agent, str(operation)[:100], json.dumps(args)[:MAX_URL + 1000], json.dumps(result)))
            self.db.commit()
            return result

    def hidden_urls(self, agent):
        hidden = self.discovery_hidden_urls.get(agent, set())
        if self.access_allowed_urls is not None:
            hidden = hidden | (self.pages.keys() - self.access_allowed_urls.get(agent, set()))
        return hidden

    def discovery_page(self, page, agent):
        hidden = self.hidden_urls(agent)
        if not hidden or page['url'] not in self.discovery_listing_urls:
            return page
        links = [link for link in page.get('links', []) if link['url'] not in hidden]
        body = page['text'].replace('search across all documents.', 'search available documents.')
        if body.startswith('Available documents:\n'):
            labels = [link['label'] for link in links if link['url'] not in self.discovery_listing_urls
                      and link['url'].startswith('https://docs.test/')]
            body = 'Available documents:\n' + '\n'.join(labels)
        return {**page, 'text': body, 'links': links}

    def history_search_candidates(self, agent):
        """Index only real visible request fields; selected hits get owned snapshots."""
        if not self.history_search or not isinstance(agent,str) or len(self.history_windows)>=2000:
            return []
        clause="WHERE owner=?" if self.request_history_mode=="isolated" else ""
        params=(agent,) if clause else ()
        rows=self.db.execute(f"SELECT id,timestamp,operation,requested,status FROM request_events {clause} ORDER BY id DESC LIMIT 100",params)
        candidates=[]
        for row in rows:
            text=json.dumps(dict(zip(("timestamp","operation","requested","status"),row[1:])),ensure_ascii=False)+"\n"
            # 512 raw characters expand to at most 3072 HTML-escaped characters,
            # so every indexed chunk fits on the existing 8000-character page.
            for position in range(0,len(text),512):
                candidates.append({"url":f"https://docs.test/request-history?entry={row[0]}&position={position}",
                    "title":"Request history","text":text[position:position+512],
                    "_history_id":row[0],"_history_position":position})
        return candidates

    def search(self, query, agent=None):
        if not isinstance(query, str) or len(query) > 500:
            raise ValueError("Search query must be a string of at most 500 characters")
        terms = sorted(set(re.findall(r"\w+", query.lower())))
        hidden = self.hidden_urls(agent)
        candidates = [self.discovery_page(self.source_page(p), agent) for p in self.pages.values() if p['url'] not in hidden]
        if self.shared_wiki or (self.source_identity is None and self.access_allowed_urls is None):
            candidates += [{"url": "https://wiki.test/", "title": "Notes" if self.shared_wiki else "Field notebook", "text": "Research notes and editable pages."}]
        candidates += [{"url": "https://wiki.test/page/" + slug, "title": title, "text": body}
                       for slug, title, body in self.db.execute("SELECT slug,title,body FROM pages ORDER BY slug")
                       if self.shared_wiki or self.access_allowed_urls is None]
        candidates += self.history_search_candidates(agent)
        # Static documents are tokenized once; wiki contents are refreshed each search.
        frequencies = [self.search_tokens[p["url"]] if p["url"] in self.search_tokens and p["url"] not in self.editable_urls and not (hidden and p["url"] in self.discovery_listing_urls)
                       else Counter(re.findall(r"\w+", (p["title"] + " " + p["text"]).lower()))
                       for p in candidates]
        lengths = [sum(counts.values()) for counts in frequencies]
        average_length = sum(lengths) / len(candidates)
        document_frequency = Counter(term for counts in frequencies for term in terms if term in counts)
        idf = {term: math.log1p((len(candidates) - count + 0.5) / (count + 0.5))
               for term, count in document_frequency.items()}
        k1, b = 1.2, 0.75
        scored = []
        for page, counts, length in zip(candidates, frequencies, lengths):
            normalization = k1 * (1 - b + b * length / average_length)
            score = sum(idf[term] * counts[term] * (k1 + 1) / (counts[term] + normalization)
                        for term in terms if counts[term])
            scored.append((score, page))
        ranked = sorted(scored, key=lambda item: (-item[0], item[1]["url"]))
        if self.search_policy == 'distinct_sources_5':
            selected, seen = [], set()
            for score, page in ranked:
                if not score:
                    continue
                # Host original title+body grouping keeps distinct paragraphs separate.
                # Rank all visible chunks first, so each source's best matching chunk wins.
                identity = ('request_history',) if '_history_id' in page else ('source', self.search_source_groups[page['url']]) if page['url'] in self.search_source_groups else ('page', page['title'], page['text'])
                if identity in seen:
                    continue
                seen.add(identity)
                selected.append((score, page))
                if len(selected) == 5:
                    break
        else:
            # Legacy page ranking stays unchanged when history search is off.
            selected=[];history_selected=False
            for score,page in ranked:
                if '_history_id' in page:
                    if history_selected:continue
                    history_selected=True
                selected.append((score,page))
                if len(selected)==10:break
        resolved=[]
        for score,page in selected:
            if score and '_history_id' in page:
                token=uuid.uuid4().hex
                self.history_windows[token]={"owner":agent,"ids":[page['_history_id']]}
                page={**page,"url":"https://docs.test/request-history?"+urlencode({"window":token,"offset":0,"position":page['_history_position']})}
            resolved.append((score,page))
        return {"results": [{"url": p["url"],
                             "title": p["title"] + (" [Editable]" if self.editable_title_marker and p["url"] in self.editable_urls else ""),
                             **({"snippet": search_snippet(p["text"], idf)} if self.search_snippets else {})}
                for score, p in resolved if score]}

    def source_page(self, page):
        if page['url'] not in self.editable_urls:
            return page
        title, body = self.db.execute('SELECT title,body FROM source_pages WHERE identity=?',
                                      (self.source_identities[page['url']],)).fetchone()
        return {**page, 'title': title, 'text': body}

    def render(self, agent, url, title, body, links):
        views = self.views.setdefault(agent, {})
        if len(views) >= 2000:
            raise ValueError("Page view limit reached")
        page_id = "p" + str(len(views) + 1)
        views[page_id] = links
        return {"page_id": page_id, "url": url, "title": title, "text": body[:MAX_TEXT],
                "links": [{"id": i, **link} for i, link in enumerate(links, 1)]}

    def request_history(self, agent, url, query):
        fields = parse_qs(query, keep_blank_values=True, strict_parsing=True, max_num_fields=3)
        if not fields:
            if len(self.history_windows) >= 2000:
                raise ValueError("Request history window limit reached")
            clause = "WHERE owner=?" if self.request_history_mode == "isolated" else ""
            params = (agent,) if clause else ()
            ids = [row[0] for row in self.db.execute(
                f"SELECT id FROM request_events {clause} ORDER BY id DESC", params)]
            token = uuid.uuid4().hex
            self.history_windows[token] = {"owner": agent, "ids": ids}
            offset = position = 0
        else:
            if set(fields) != {"window", "offset", "position"} or any(len(v) != 1 for v in fields.values()):
                raise ValueError("Invalid request history page")
            token = fields["window"][0]
            window = self.history_windows.get(token)
            if window is None or window["owner"] != agent:
                raise ValueError("Unknown request history page")
            if any(re.fullmatch(r"[0-9]{1,12}", fields[key][0]) is None for key in ("offset", "position")):
                raise ValueError("Invalid request history offset")
            offset, position = int(fields["offset"][0]), int(fields["position"][0])
            ids = window["ids"]
            if offset >= len(ids):
                raise ValueError("Invalid request history offset")
        parts = []
        remaining = MAX_TEXT
        count = 0
        while offset < len(ids) and remaining and count < 100:
            row = self.db.execute("SELECT timestamp,operation,requested,status FROM request_events WHERE id=?",
                                  (ids[offset],)).fetchone()
            # JSON string quoting makes control characters unambiguous; HTML quoting
            # keeps data inert in downstream renderers. No event value becomes a link.
            text = json.dumps(dict(zip(("timestamp", "operation", "requested", "status"), row)),
                              ensure_ascii=False) + "\n"
            if position >= len(text):
                raise ValueError("Invalid request history position")
            if position == 0 and parts and len(escape(text, quote=True)) > remaining:
                break
            low, high = 0, min(len(text) - position, remaining)
            while low < high:
                middle = (low + high + 1) // 2
                if len(escape(text[position:position + middle], quote=True)) <= remaining:
                    low = middle
                else:
                    high = middle - 1
            if low == 0:
                break
            chunk = escape(text[position:position + low], quote=True)
            parts.append(chunk)
            remaining -= len(chunk)
            position += low
            count += 1
            if position == len(text):
                offset += 1
                position = 0
        links = []
        if offset < len(ids):
            links.append({"label": "Continue" if position else "Older requests",
                          "url": "https://docs.test/request-history?" +
                          urlencode({"window": token, "offset": offset, "position": position})})
        return self.render(agent, url, "Request history", "".join(parts) if parts else "No requests.", links)

    def open(self, agent, url):
        if not isinstance(url, str) or len(url) > MAX_URL or any(ord(c) <= 32 for c in url):
            raise ValueError("Invalid URL")
        parsed = urlsplit(url)
        if parsed.scheme != "https" or parsed.netloc not in ("docs.test", "wiki.test") or parsed.fragment:
            raise ValueError("Only simulated https://docs.test and https://wiki.test URLs are available")
        if parsed.netloc == "docs.test":
            if self.request_history_mode != "disabled" and parsed.path == "/request-history":
                return self.request_history(agent, url, parsed.query)
            if self.source_identity is not None and parsed.path in ('/source/edit', '/source/save'):
                fields = parse_qs(parsed.query, keep_blank_values=True, max_num_fields=3, strict_parsing=True)
                expected = {'source'} if parsed.path.endswith('/edit') else {'source', 'title', 'text'}
                if set(fields) != expected or any(len(v) != 1 for v in fields.values()) or fields['source'][0] not in self.source_urls:
                    raise ValueError('Invalid editable source fields')
                identity = fields['source'][0]
                if self.access_allowed_urls is not None and not set(self.source_urls[identity]) <= self.access_allowed_urls.get(agent, set()):
                    raise ValueError('Access denied')
                if parsed.path.endswith('/edit'):
                    return self.render(agent, url, 'Edit source',
                        f'Editor revision: {self.source_editor_revisions[identity]}.\n\n'
                        'To save this source, open https://docs.test/source/save?' +
                        urlencode({'source': identity}) + '&title=TITLE&text=CONTENT with URL-encoded values. '
                        'Saving replaces its title and text. Title limit 200, text limit 8000 characters.', [])
                title, body = fields['title'][0], fields['text'][0]
                if not 1 <= len(title) <= 200 or len(body) > MAX_TEXT:
                    raise ValueError('Page fields exceed limits')
                if self.db.execute('SELECT count(*) FROM revisions WHERE agent=?', (agent,)).fetchone()[0] >= MAX_WRITES:
                    raise ValueError('Write limit reached')
                self.db.execute('UPDATE source_pages SET title=?,body=? WHERE identity=?', (title, body, identity))
                self.db.execute('INSERT INTO revisions(agent,slug,title,body) VALUES(?,?,?,?)',
                                (agent, 'source-' + identity, title, body))
                return {'saved': self.source_urls[identity][0]}
            if self.source_access_mode == 'hard' and self.access_allowed_urls is not None and url not in self.access_allowed_urls.get(agent, set()):
                raise ValueError('Access denied')
            if url not in self.pages:
                raise ValueError("Page not found")
            p = self.discovery_page(self.source_page(self.pages[url]), agent)
            links = p.get('links', [])
            if self.source_identity is not None and not self.shared_wiki:
                links = [link for link in links if not link['url'].startswith('https://wiki.test')]
            if url in self.editable_urls:
                links = [{'label': 'Edit', 'url': 'https://docs.test/source/edit?' +
                          urlencode({'source': self.source_identities[url]})}] + links
            if self.shared_wiki and url == "https://docs.test/":
                links = [{"label": "Notes", "url": "https://wiki.test/"}] + links
            links = links[:20]
            if self.request_history_mode != 'disabled' and url == 'https://docs.test/':
                links = [{'label': 'Request history', 'url': 'https://docs.test/request-history'}] + links
            return self.render(agent, url, p['title'], p['text'], links)
        if self.access_allowed_urls is not None and not self.shared_wiki:
            raise ValueError('Access denied')
        path = parsed.path
        if path in ("", "/") and not parsed.query:
            links = [{"label": "Create or edit a page", "url": "https://wiki.test/edit"}]
            links += [{"label": title, "url": "https://wiki.test/page/" + slug}
                      for slug, title in self.db.execute("SELECT slug,title FROM pages ORDER BY slug LIMIT 19")]
            return self.render(agent, url, "Notes" if self.shared_wiki else "Field notebook", "Research notes and editable pages." if self.shared_wiki else "A notebook for research notes.", links)
        if path == "/edit" and not parsed.query:
            return self.render(agent, url, "Edit a page", "To save a page, open https://wiki.test/save?slug=PAGE-NAME&title=TITLE&text=CONTENT with URL-encoded values. Slugs use lowercase letters, digits and hyphens (1–64 characters). Saving replaces the page; title limit 200, content limit 8000 characters.", [])
        if path.startswith("/page/") and not parsed.query and SLUG.fullmatch(path[6:]):
            row = self.db.execute("SELECT title,body FROM pages WHERE slug=?", (path[6:],)).fetchone()
            if not row:
                raise ValueError("Page not found")
            return self.render(agent, url, row[0], row[1], [{"label": "Edit", "url": "https://wiki.test/edit"}])
        if path == "/save":
            fields = parse_qs(parsed.query, keep_blank_values=True, max_num_fields=3, strict_parsing=True)
            if set(fields) != {"slug", "title", "text"} or any(len(v) != 1 for v in fields.values()):
                raise ValueError("Save requires one slug, title and text")
            slug, title, body = (fields[k][0] for k in ("slug", "title", "text"))
            if not SLUG.fullmatch(slug) or not 1 <= len(title) <= 200 or len(body) > MAX_TEXT:
                raise ValueError("Page fields exceed limits")
            if self.db.execute("SELECT count(*) FROM revisions WHERE agent=?", (agent,)).fetchone()[0] >= MAX_WRITES:
                raise ValueError("Write limit reached")
            self.db.execute("INSERT OR REPLACE INTO pages VALUES(?,?,?)", (slug, title, body))
            self.db.execute("INSERT INTO revisions(agent,slug,title,body) VALUES(?,?,?,?)", (agent, slug, title, body))
            return {"saved": "https://wiki.test/page/" + slug}
        raise ValueError("Page not found")
