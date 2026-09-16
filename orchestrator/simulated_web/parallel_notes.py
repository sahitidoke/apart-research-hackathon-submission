"""Mandatory freeform-note semantics for completion-driven independent workers."""
from orchestrator.simulated_web.mandatory_notes import APPEND_INSTRUCTION, ATTEMPTS, ATTEMPT_TOKENS, SECONDS

INSTRUCTION = APPEND_INSTRUCTION.replace('before the schedule advances', 'before you advance to your next question').replace('before allowing the next scheduled phase', 'before allowing you to advance to your next question')


def validate_entry(message, metadata):
    content = message.get('content')
    if (metadata.get('done_reason') == 'length' or metadata.get('final_only_contract_violation')
            or message.get('tool_calls') or message.get('thinking') or not isinstance(content, str)
            or not content.strip() or len(content) > 6000):
        raise ValueError('Require a complete nonblank freeform entry of at most6000characters without thinking or tools')
    return content


class PersistenceEvents:
    def __init__(self, events):
        self.events = events

    def emit(self, event, **fields):
        return self.events.emit(event, actor='host_notebook_persistence', **fields)


def host_call(browser, agent, operation, args):
    original = browser.events
    browser.events = PersistenceEvents(original)
    try:
        return browser.call(agent, operation, args)
    finally:
        browser.events = original


def persist_entry(browser, agent, content, note, save_artifact):
    """Runs only on central authority; direct Unicode-safe append, exact read/DB check."""
    saved = None
    for operation in ('append', 'verify'):
        before = browser.db.execute('SELECT coalesce(max(id),0) FROM request_events').fetchone()[0]
        result = (host_call(browser, agent, 'append_notebook', {'text': content}) if operation == 'append'
                  else host_call(browser, agent, 'read_notebook', {'url': saved, 'revision': ''}))
        action = {'actor': 'host_notebook_persistence', 'operation': operation, 'response': result,
                  'request_event_ids': [r[0] for r in browser.db.execute('SELECT id FROM request_events WHERE id>?', (before,))]}
        note['host_persistence_actions'].append(action)
        save_artifact()
        if 'error' in result:
            raise RuntimeError('Host notebook persistence failed: ' + str(result['error']))
        if operation == 'append':
            saved = result.get('saved', '')
            root = 'https://wiki.test/page/' + browser.notebooks[agent] + '-entry-'
            suffix = saved.removeprefix(root)
            if not saved.startswith(root) or len(suffix) != 6 or not suffix.isdigit():
                raise RuntimeError('Unexpected owned notebook entry URL')
            note['saved_url'] = saved
            note['saved_revision'] = result.get('revision')
        elif result.get('text') != content or result.get('revision') != note['saved_revision'] or result.get('author') != browser.visible_labels[agent]:
            raise RuntimeError('Exact notebook read verification failed')
    slug = saved.removeprefix('https://wiki.test/page/')
    if browser.db.execute('SELECT body FROM pages WHERE slug=?', (slug,)).fetchone() != (content,):
        raise RuntimeError('Exact notebook database verification failed')
    note.update(status='saved', persistence_verified=True, text=content)
    save_artifact()
    return saved
