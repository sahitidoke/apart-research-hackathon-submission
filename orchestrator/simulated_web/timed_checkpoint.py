"""Read-only validation/loading of timed host checkpoints; no model execution."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

from orchestrator.simulated_web.sqlite_snapshot import snapshot_connection
from orchestrator.simulated_web.browser import Browser
from orchestrator.simulated_web.timed_policy import TimedPolicy
from orchestrator.simulated_web.private_memory import PrivateScratchpad


CHECKPOINT_FILES = {'settings.json', 'dataset.json', 'pages.json', 'results.json',
                    'transitions.json', 'history.json', 'browser.json', 'wiki.sqlite3'}


@dataclass
class LoadedCheckpoint:
    path: Path
    manifest: dict
    data: dict
    browser: Browser
    manifest_sha256: str

    def close(self):
        self.browser.close()

    @property
    def complete(self):
        return self.manifest['session_complete']


def load_checkpoint(path):
    """Return verified host state with a private in-memory browser. Caller must close().

    Works for complete checkpoints too, without model calls or persistent writes.
    Hashes detect accidental corruption; they are not a signature/authentication.
    """
    path = Path(path).resolve()
    manifest_path = path / 'checkpoint.json'
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError('Missing or unsafe checkpoint manifest')
    raw_manifest = manifest_path.read_bytes()
    manifest = json.loads(raw_manifest)
    expected_files = CHECKPOINT_FILES | ({'private-scratchpad.json'} if manifest.get('schema') == 'timed-host-checkpoint-v2' else set())
    expected_boundary = ('after_reflection_and_private_reset' if manifest.get('schema') == 'timed-host-checkpoint-v2'
                         else 'after_reflection_and_any_scheduled_compaction')
    if (manifest.get('schema') not in ('timed-host-checkpoint-v1', 'timed-host-checkpoint-v2') or manifest.get('status') != 'complete'
            or manifest.get('boundary') != expected_boundary
            or set(manifest.get('files_sha256', {})) != expected_files):
        raise ValueError('Unsupported or incomplete checkpoint schema')
    data = {}
    for name in sorted(expected_files):
        source = path / name
        if not source.is_file() or source.is_symlink():
            raise ValueError(f'Missing or unsafe checkpoint file: {name}')
        raw = source.read_bytes()
        if hashlib.sha256(raw).hexdigest() != manifest['files_sha256'][name]:
            raise ValueError(f'Checkpoint hash mismatch: {name}')
        data[name] = raw if name.endswith('.sqlite3') else json.loads(raw)
    settings = data['settings.json']
    policy = TimedPolicy(**settings['policy'])
    if (policy.memory_mode == 'private_scratchpad') != ('private-scratchpad.json' in data):
        raise ValueError('Checkpoint memory policy/schema mismatch')
    history = data['history.json']
    if (settings.get('protocol') != 'germanwiki-timed-v3' or not isinstance(history, list) or not history
            or history[0] != {'role': 'system', 'content': settings.get('system_prompt')}
            or any(not isinstance(m, dict) or m.get('role') not in ('system', 'user', 'assistant', 'tool') for m in history)
            or any(m.get('role') == 'system' for m in history[1:])):
        raise ValueError('Invalid checkpoint history or protocol')
    if policy.memory_mode == 'private_scratchpad':
        scratchpad = PrivateScratchpad.restore(data['private-scratchpad.json'], policy.scratchpad_tokens)
        topic = settings.get('topic')
        if not isinstance(topic, str) or not 1 <= len(topic.strip()) <= 2000:
            raise ValueError('Invalid checkpoint topic')
        if history != scratchpad.reset_history(history[0], topic):
            raise ValueError('Checkpoint private history is not the completed reset boundary')
    records = data['dataset.json']
    if hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest() != settings.get('dataset_sha256'):
        raise ValueError('Checkpoint dataset identity mismatch')
    count = settings['policy']['question_count']
    completed = manifest.get('completed_questions')
    if (type(count) is not int or type(completed) is not int or not 0 < completed <= count
            or completed % 5 or type(manifest.get('session_complete')) is not bool
            or manifest['session_complete'] != (completed == count)
            or manifest.get('next_phase_index') != 2 * completed + 1):
        raise ValueError('Invalid checkpoint progress boundary')
    order = settings['schedule']['orders']['agent-1']
    if (not isinstance(order, list) or len(order) != count or len(set(order)) != count
            or set(order) != {r['id'] for r in records}):
        raise ValueError('Invalid checkpoint question order')
    next_qid = order[completed] if completed < count else None
    if (manifest.get('next_question_id') != next_qid
            or manifest.get('next_phase') != ('answer' if next_qid is not None else None)):
        raise ValueError('Invalid checkpoint next phase')
    rows = data['results.json']
    transitions = data['transitions.json']
    if not isinstance(rows, list) or len(rows) != 2 * completed + 1 or len(transitions) != len(rows):
        raise ValueError('Invalid checkpoint phase history length')
    expected = [('preparation', None)]
    for qid in order[:completed]:
        expected.extend([('answer', qid), ('reflection', qid)])
    for row, transition, (phase, qid) in zip(rows, transitions, expected):
        if (row.get('phase') != phase or row.get('question_id') != qid
                or row.get('status') in ('error', 'context_exhausted')
                or transition.get('phase') != phase or transition.get('question_id') != qid):
            raise ValueError('Invalid checkpoint phase history')
    state = data['browser.json']
    if any(state.get(key) != settings.get(key) for key in ('search_snippets', 'editable_title_marker')):
        raise ValueError('Checkpoint browser configuration mismatch')
    if state.get('request_history_mode', 'disabled') != policy.request_history_mode:
        raise ValueError('Checkpoint request history mode mismatch')
    browser = Browser(data['pages.json'], ':memory:', editable_sources=settings['editable_source'],
                      search_snippets=state['search_snippets'], editable_title_marker=state['editable_title_marker'],
                      request_history_mode=policy.request_history_mode)
    try:
        for key in ('source_identities', 'source_urls', 'source_identity', 'source_editor_revisions'):
            if state.get(key) != getattr(browser, key):
                raise ValueError(f'Checkpoint browser identity mismatch: {key}')
        if state.get('editable_urls') != sorted(browser.editable_urls):
            raise ValueError('Checkpoint editable URLs mismatch')
        views = state.get('views')
        if not isinstance(views, dict) or set(views) - {'agent-1'}:
            raise ValueError('Invalid checkpoint browser agents')
        for entries in views.values():
            if (not isinstance(entries, dict) or len(entries) > 2000
                    or set(entries) != {f'p{i + 1}' for i in range(len(entries))}):
                raise ValueError('Invalid checkpoint browser page IDs')
            for links in entries.values():
                if not isinstance(links, list) or any(not isinstance(link, dict)
                        or not isinstance(link.get('url'), str) or not isinstance(link.get('label'), str) for link in links):
                    raise ValueError('Invalid checkpoint browser links')
        # Restore already-hashed bytes into a private clone; never mutate the parent.
        with snapshot_connection(data['wiki.sqlite3']) as saved:
            # Historical snapshots predate public request events. Upgrade only this private clone.
            if 'request_history_mode' not in state:
                saved.execute("CREATE TABLE request_events(id INTEGER PRIMARY KEY, owner TEXT, timestamp TEXT, operation TEXT, requested TEXT, status TEXT)")
                saved.commit()
            schema_query = "SELECT type,name,sql FROM sqlite_master ORDER BY type,name"
            if saved.execute(schema_query).fetchall() != browser.db.execute(schema_query).fetchall():
                raise ValueError('Unexpected checkpoint SQLite schema')
            if saved.execute('PRAGMA integrity_check').fetchall() != [('ok',)]:
                raise ValueError('Checkpoint SQLite integrity check failed')
            if {row[0] for row in saved.execute('SELECT identity FROM source_pages')} != set(browser.source_urls):
                raise ValueError('Checkpoint SQLite source identity mismatch')
            saved.backup(browser.db)
        windows = state.get('history_windows', {})
        if not isinstance(windows, dict) or len(windows) > 2000:
            raise ValueError('Invalid checkpoint request history windows')
        for token, window in windows.items():
            if (re.fullmatch(r'[0-9a-f]{32}', token) is None or not isinstance(window, dict)
                    or set(window) != {'owner', 'ids'} or window['owner'] != 'agent-1'
                    or not isinstance(window['ids'], list)
                    or any(type(i) is not int or i <= 0 for i in window['ids'])
                    or window['ids'] != sorted(set(window['ids']), reverse=True)):
                raise ValueError('Invalid checkpoint request history window')
            for event_id in window['ids']:
                row = browser.db.execute('SELECT owner FROM request_events WHERE id=?', (event_id,)).fetchone()
                if row is None or (policy.request_history_mode == 'isolated' and row[0] != window['owner']):
                    raise ValueError('Invalid checkpoint request history event reference')
        browser.history_windows = windows
        browser.views = views
        return LoadedCheckpoint(path, manifest, data, browser, hashlib.sha256(raw_manifest).hexdigest())
    except BaseException:
        browser.close()
        raise


def checkpoint_model_digest(checkpoint):
    model = checkpoint.data['settings.json'].get('provenance', {}).get('model', {})
    digest = model.get('digest')
    if not isinstance(digest, str) or re.fullmatch(r'(sha256:)?[0-9a-f]{64}', digest) is None:
        raise ValueError('Resuming requires a saved verified model digest')
    return digest
