"""Host-only browser observation retention at completed question boundaries."""
from orchestrator.simulated_web.browser import TOOLS
from orchestrator.simulated_web.musique_batch import write_json

BROWSER_TOOLS = frozenset(tool['function']['name'] for tool in TOOLS)
OMITTED = '[Browser response omitted after reflection.]'
ANSWER_OMITTED = '[Browser response omitted after answer.]'
SELECTIVE_NOTICE = ('Browser responses remain available through preparation and the first answer and reflection. '
                    'After each reflection, their contents are omitted from your private conversation; '
                    'your own messages and tool requests remain available until any later memory reset.')


def retention_notice(policy):
    if policy.browser_retention == 'question_boundary':
        return SELECTIVE_NOTICE + (' Your private conversation may also be summarized between questions.'
                                   if policy.compaction_enabled else '')
    return ('Your private conversation may be summarized between questions.' if policy.compaction_enabled
            else 'Your full conversation is retained.')


def retain_at_question_boundary(history, policy, run_dir, phase_index, agent, question_id, boundary="after_reflection"):
    """Mask only browser tool contents, retaining call/result structure and raw evidence.

    No inference, tokenization or warmup is performed here. Subsequent transport
    preflight supplies native context accounting using the retained history.
    """
    if boundary not in ('after_answer', 'after_reflection'):
        raise ValueError('Invalid observation retention boundary')
    omitted = ANSWER_OMITTED if boundary == 'after_answer' else OMITTED
    if policy.browser_retention == 'full':
        return
    replacements = [(index, message) for index, message in enumerate(history)
                    if message.get('role') == 'tool' and message.get('tool_name') in BROWSER_TOOLS
                    and message.get('content') not in (OMITTED, ANSWER_OMITTED)]
    snapshot = f'history-before-retention-{phase_index:02d}-{agent}.json'
    write_json(run_dir / snapshot, history)
    characters = sum(len(message.get('content', '')) for _, message in replacements)
    metrics = {'policy': policy.browser_retention, 'boundary': boundary,
               'phase_index': phase_index, 'agent': agent, 'question_id': question_id,
               'previous_history_path': snapshot, 'removed_message_contents': len(replacements),
               'removed_content_characters': characters,
               'replacement_content_characters': len(omitted) * len(replacements),
               'messages_deleted': 0, 'generation_performed': False,
               'token_accounting': 'No retention token estimate; subsequent native preflight counts retained context.'}
    # Archive evidence before modifying active history; other agents are never supplied here.
    write_json(run_dir / f'browser-retention-{phase_index:02d}-{agent}.json', metrics)
    for index, message in replacements:
        history[index] = {**message, 'content': omitted}
    print(f'{agent} browser retention: omitted {len(replacements)} response contents '
          f'({characters} characters) {boundary.replace('_', ' ')}; raw snapshot={snapshot}', flush=True)
