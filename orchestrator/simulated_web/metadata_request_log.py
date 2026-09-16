"""Opt-in public request metadata: URLs/titles/status, never response or note bodies."""
from datetime import datetime, timezone
import json
from urllib.parse import urlsplit, urlunsplit

from orchestrator.simulated_web.async_notebooks import AsyncNotebookBrowser
from orchestrator.simulated_web.source_access import access_browser_options
from orchestrator.simulated_web.source_discovery import browser_discovery
from orchestrator.simulated_web.token_pair import search_browser_options


def public_url(value):
    if not isinstance(value, str):
        return None
    try:
        parsed = urlsplit(value)
    except ValueError:
        return None
    if parsed.scheme != 'https' or parsed.netloc not in ('docs.test', 'wiki.test'):
        return None
    # Query strings may contain legacy save bodies or owned history tokens.
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, '', ''))


class MetadataLogBrowser(AsyncNotebookBrowser):
    def call(self, agent, operation, args):
        with self.lock:
            before = self.db.execute('SELECT coalesce(max(id),0) FROM request_events').fetchone()[0]
            result = super().call(agent, operation, args)
            metadata = {}
            url = public_url(result.get('saved') or result.get('url'))
            if url is None and isinstance(args, dict):
                url = public_url(args.get('url'))
            if url:
                metadata['url'] = url
            title = result.get('title')
            if result.get('saved'):
                row = self.db.execute('SELECT title FROM pages WHERE slug=?', (urlsplit(result['saved']).path.removeprefix('/page/'),)).fetchone()
                title = row[0] if row else None
            if isinstance(title, str):
                metadata['title'] = title[:500]
            requested = json.dumps(metadata, ensure_ascii=False)
            ids = [r[0] for r in self.db.execute('SELECT id FROM request_events WHERE id>?', (before,))]
            if ids:
                self.db.execute('UPDATE request_events SET requested=? WHERE id>?', (requested, before))
            else:
                self.db.execute('INSERT INTO request_events(owner,timestamp,operation,requested,status) VALUES(?,?,?,?,?)',
                    (agent, datetime.now(timezone.utc).isoformat(), operation, requested, 'error' if 'error' in result else 'success'))
            self.db.commit()
            return result


def make_metadata_browser(settings, pages, path, events):
    return MetadataLogBrowser(pages, path, events=events, append_notes=True,
        notebooks=settings['notebooks'], neutral_notebook=True, visible_labels=settings['visible_labels'],
        notebook_tools=True, editable_sources=[], editable_title_marker=False, search_snippets=False,
        request_history_mode='shared', history_search=True, **browser_discovery(settings['discovery_plan']),
        **search_browser_options(settings['discovery_plan'], 'distinct_sources_5'),
        **access_browser_options(settings['access_plan']))
