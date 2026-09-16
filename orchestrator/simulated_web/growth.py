"""Conservative per-phase growth ledger and structured browser-result excerpts."""
import copy
import json


def result_bytes(result):
    """Bytes of the exact ASCII-escaped JSON content delivered by the runner."""
    return len(json.dumps(result).encode('utf-8'))


def bounded_result(result, allowance):
    """Preserve source handles and valid JSON; return None if metadata cannot fit.

    Text fields may become excerpts; lists may lose trailing entries, retaining
    original IDs/URLs for every surviving link/result. Never slice serialized JSON.
    """
    if result_bytes(result) <= allowance:
        return result
    fallback = {'omitted': 'Result body omitted by phase growth budget; source arguments remain in the tool call.',
                'status': ('saved' if isinstance(result, dict) and 'saved' in result else
                           'error' if isinstance(result, dict) and 'error' in result else 'completed')}
    if isinstance(result, dict):
        for key in ('page_id', 'url', 'saved'):
            if key in result and result_bytes({**fallback, key: result[key]}) <= allowance:
                fallback[key] = result[key]
    if result_bytes(fallback) > allowance:
        return None
    if not isinstance(result, dict):
        return fallback
    candidate = copy.deepcopy(result)
    candidate['omitted'] = 'Result excerpt: text or trailing entries omitted by phase growth budget.'
    text_fields = []
    lists = []

    def collect(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key in ('text', 'snippet', 'title', 'label') and isinstance(child, str):
                    text_fields.append((value, key, child))
                elif isinstance(child, (list, dict)):
                    collect(child)
        elif isinstance(value, list):
            lists.append(value)
            for child in value:
                collect(child)
    collect(candidate)
    for obj, key, original in text_fields:
        obj[key] = ''
    # If URL/link metadata alone is large, omit whole trailing entries, not IDs/URLs.
    while result_bytes(candidate) > allowance:
        nonempty = [items for items in lists if items]
        if not nonempty:
            return fallback
        max(nonempty, key=result_bytes).pop()
    low, high = 0, max((len(original) for _, _, original in text_fields), default=0)
    while low < high:
        middle = (low + high + 1) // 2
        for obj, key, original in text_fields:
            obj[key] = original[:middle]
        if result_bytes(candidate) <= allowance:
            low = middle
        else:
            high = middle - 1
    for obj, key, original in text_fields:
        obj[key] = original[:low]
    return candidate
