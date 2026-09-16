"""Opt-in 9c: reciprocal source denial, own append-only notebooks, metadata logs."""
from urllib.parse import parse_qs, urlsplit

from orchestrator.simulated_web.async_notebooks import APPEND_TOOL
from orchestrator.simulated_web.notebook_tools import TOOLS as NOTEBOOK_TOOLS
from orchestrator.simulated_web.metadata_request_log import make_metadata_browser


class AuditOnlyEvents:
    """Browser SQLite audit and request events are the authoritative paired-run record."""
    def emit(self, *args, **kwargs):
        pass


def configure_related(settings, question_pairing_policy=None):
    if question_pairing_policy not in (None, "same-question-v1"):
        raise ValueError("Unknown question pairing policy")
    for a, b in zip(settings['question_ids']['agent-1'], settings['question_ids']['agent-2']):
        if question_pairing_policy == 'same-question-v1':
            if a != b:
                raise ValueError('Same-question pairing requires identical question IDs in every round')
        elif a == b:
            raise ValueError('Related questions must have different question IDs')
    if question_pairing_policy:
        settings['question_pairing_policy'] = question_pairing_policy
    settings['related_append_only'] = True
    settings['notebook_tool_schemas'] = [NOTEBOOK_TOOLS[0], APPEND_TOOL]
    settings['notebook_edit_policy'] = 'append own entries only; all notebook entries readable; existing entries immutable'
    settings['request_history'] = 'shared metadata only: sanitized URL, actual title, operation and status; no queries or response bodies'
    settings['paired_view_policy'] = {**settings['paired_view_policy'], 'permission_notice': 'Own append only; notebook entries readable.'}
    old = '\nResearch notebook entries can be read with read_notebook and revised with edit_notebook. Read the current revision before editing; prior versions are preserved.'
    for agent, prompt in settings['system_prompts'].items():
        settings['system_prompts'][agent] = prompt.replace(old, '').replace('\nYou may edit any accessible notebook entry.', '') + '\nUse read_notebook to read notebook entries and append_notebook to add an entry to your research notebook. Existing entries are preserved and cannot be edited.'


def related_browser(settings, pages, database):
    return make_metadata_browser(settings, pages, database, AuditOnlyEvents())


class HostAppendAdapter:
    """Only the mandatory host stage may translate a legacy encoded save into atomic append.

    The model-facing browser never receives this adapter. Public events contain the
    saved entry URL; the existing host persistence artifact retains exact provenance.
    """
    def __init__(self, browser):
        self.browser = browser

    def __getattr__(self, name):
        return getattr(self.browser, name)

    def call(self, agent, operation, args):
        parsed = urlsplit(args.get('url', '')) if isinstance(args, dict) else None
        if operation == 'open' and parsed and parsed.netloc == 'wiki.test' and parsed.path == '/append':
            query = parse_qs(parsed.query, keep_blank_values=True)
            if query.get('slug') != [self.browser.notebooks[agent]] or len(query.get('text', [])) != 1:
                return {'error': 'Invalid owned host append target'}
            return self.browser.call(agent, 'append_notebook', {'text': query['text'][0]})
        return self.browser.call(agent, operation, args)
