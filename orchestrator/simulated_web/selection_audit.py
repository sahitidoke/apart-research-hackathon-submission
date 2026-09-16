"""Offline document selection diagnostic on recorded session browser events."""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
from urllib.parse import urlsplit

from orchestrator.simulated_web.search_replay import require, support_index


def fraction(numerator, denominator):
    return {'numerator': numerator, 'denominator': denominator,
            'rate': numerator / denominator if denominator else None}


def selection_metrics(opened, shown, support):
    opened_supports = set(opened) & support
    already = {key for key in shown if key in opened and min(opened[key]) < shown[key]}
    later = {key for key in shown if any(event > shown[key] for event in opened.get(key, []))}
    eligible = set(shown) - already
    never = set(shown) - set(opened)
    return {'opened_document_precision': fraction(len(opened_supports), len(opened)),
            'opened_support_recall': fraction(len(opened_supports), len(support)),
            'shown_support_then_subsequently_opened': fraction(len(later), len(shown)),
            'newly_shown_support_selection': fraction(len(later & eligible), len(eligible)),
            'shown_support_count': len(shown), 'already_opened_before_first_exposure': len(already),
            'already_opened_and_reopened_after_exposure': len(already & later),
            'shown_but_never_opened': len(never),
            'all_support_opened': bool(support) and opened_supports == support}


def aggregate(assignments):
    summary = {'assignment_count': len(assignments),
               'zero_search_assignments': sum(a['search_calls'] == 0 for a in assignments),
               'zero_document_open_assignments': sum(not a['opened_documents'] for a in assignments)}
    for key in ('opened_document_precision', 'opened_support_recall',
                'shown_support_then_subsequently_opened', 'newly_shown_support_selection'):
        summary[key] = fraction(sum(a['metrics'][key]['numerator'] for a in assignments),
                                sum(a['metrics'][key]['denominator'] for a in assignments))
    for key in ('shown_support_count', 'already_opened_before_first_exposure',
                'already_opened_and_reopened_after_exposure', 'shown_but_never_opened'):
        summary[key] = sum(a['metrics'][key] for a in assignments)
    summary['all_support_opened'] = fraction(sum(a['metrics']['all_support_opened'] for a in assignments),
                                           len(assignments))
    for key in ('search_calls', 'successful_search_calls', 'document_open_calls',
                'wiki_open_calls', 'index_open_calls', 'failed_calls'):
        summary[key] = sum(a[key] for a in assignments)
    return summary


def audit_selection(run_root):
    root = Path(run_root).resolve()
    raw = {name: (root / name).read_bytes() for name in ('dataset.jsonl', 'pages.json', 'results.json')}
    records = [json.loads(line) for line in raw['dataset.jsonl'].splitlines() if line.strip()]
    pages, results = json.loads(raw['pages.json']), json.loads(raw['results.json'])
    url_identity, supports, descriptions = support_index(records, pages)
    questions = {r['id']: r['question'] for r in records}
    document_urls = {p['url'] for p in pages}
    settings_path = root / 'settings.json'
    if settings_path.exists():
        settings = json.loads(settings_path.read_text())
        for name, suffix in (('dataset', '.jsonl'), ('pages', '.json')):
            if name + '_sha256' in settings:
                require(settings[name + '_sha256'] == hashlib.sha256(raw[name + suffix]).hexdigest(),
                        f'{name + suffix} fingerprint mismatch')
    source = sqlite3.connect((root / 'web' / 'wiki.sqlite3').as_uri() + '?mode=ro', uri=True)
    try:
        source.execute('PRAGMA query_only=ON')
        source.execute('BEGIN')
        audit = source.execute('SELECT id,agent,operation,args,result FROM audit ORDER BY id').fetchall()
    finally:
        source.close()
    require([a[0] for a in audit] == list(range(1, len(audit) + 1)), 'Audit IDs must be contiguous from 1')
    assignments, seen = [], set()
    for row in results:
        key = (row['slot'], row['agent'])
        require(key not in seen and row['id'] in supports, 'Duplicate or unknown assignment')
        seen.add(key)
        before, after = (row['wiki_boundary'][side]['audit'] for side in ('before', 'after'))
        require(type(before) is int and type(after) is int and 0 <= before <= after <= len(audit),
                'Invalid assignment audit boundary')
        assignments.append({'slot': row['slot'], 'agent': row['agent'], 'id': row['id'],
                            'question': questions[row['id']], 'wiki_boundary': row['wiki_boundary'],
                            'search_calls': 0, 'successful_search_calls': 0, 'document_open_calls': 0,
                            'wiki_open_calls': 0, 'index_open_calls': 0, 'failed_calls': 0,
                            'events': [], 'opened_documents': {}, 'shown_supports': {},
                            'wiki_urls': set(), 'index_urls': set()})
    assignments.sort(key=lambda a: (a['slot'], a['agent']))
    for audit_id, agent, operation, _, result_text in audit:
        matches = [a for a in assignments if a['agent'] == agent and
                   a['wiki_boundary']['before']['audit'] < audit_id <= a['wiki_boundary']['after']['audit']]
        require(len(matches) == 1, f'Audit {audit_id}: event must map to exactly one assignment')
        assignment = matches[0]
        result = json.loads(result_text)
        require(isinstance(result, dict), f'Audit {audit_id}: result must be an object')
        if operation == 'search':
            assignment['search_calls'] += 1
        if 'error' in result:
            assignment['failed_calls'] += 1
            continue
        support = supports[assignment['id']]
        if operation == 'search':
            urls = [p['url'] for p in result['results']]
            require(len(urls) <= 10 and len(urls) == len(set(urls)), f'Audit {audit_id}: invalid search results')
            for url in urls:
                require(url in document_urls or wiki_url(url), f'Audit {audit_id}: unknown search result URL')
            assignment['successful_search_calls'] += 1
            shown = {url_identity[url] for url in urls if url in url_identity} & support
            for key in shown:
                assignment['shown_supports'].setdefault(key, audit_id)
            assignment['events'].append({'audit_id': audit_id, 'operation': operation, 'urls': urls,
                                         'support_identities': sorted(shown)})
        else:
            require(operation in ('open', 'click'), f'Audit {audit_id}: unexpected successful operation')
            # A save acknowledgement is not an opened/read page.
            if 'saved' in result:
                continue
            url = result['url']
            require(isinstance(result.get('text'), str) and isinstance(result.get('page_id'), str),
                    f'Audit {audit_id}: successful page result lacks rendered text or page ID')
            if url in url_identity:
                key = url_identity[url]
                assignment['opened_documents'].setdefault(key, []).append(audit_id)
                assignment['document_open_calls'] += 1
                kind = 'document'
            elif url in document_urls:
                assignment['index_open_calls'] += 1
                assignment['index_urls'].add(url)
                kind = 'index'
            elif wiki_url(url):
                assignment['wiki_open_calls'] += 1
                assignment['wiki_urls'].add(url)
                kind = 'wiki'
            else:
                raise ValueError(f'Audit {audit_id}: unknown opened URL')
            assignment['events'].append({'audit_id': audit_id, 'operation': operation, 'url': url, 'kind': kind})
    skipped = []
    for assignment in assignments:
        opened, shown = assignment['opened_documents'], assignment['shown_supports']
        support = supports[assignment['id']]
        assignment['metrics'] = selection_metrics(opened, shown, support)
        assignment['support_details'] = []
        for key in sorted(support):
            exposure, opens = shown.get(key), opened.get(key, [])
            if exposure is None:
                state = 'opened_without_search_exposure' if opens else 'neither_shown_nor_opened'
            elif opens and min(opens) < exposure:
                state = 'already_opened_before_first_exposure'
            elif opens:
                state = 'opened_after_first_exposure'
            else:
                state = 'shown_but_never_opened'
            detail = {'identity': key, 'title': descriptions[key]['title'], 'first_exposure_audit_id': exposure,
                      'open_audit_ids': opens, 'state': state}
            assignment['support_details'].append(detail)
            if state == 'shown_but_never_opened':
                skipped.append({'slot': assignment['slot'], 'agent': assignment['agent'], 'id': assignment['id'],
                                'question': assignment['question'], **detail})
        for key in ('wiki_urls', 'index_urls'):
            assignment[key] = sorted(assignment[key])
    skipped.sort(key=lambda row: (row['first_exposure_audit_id'], row['agent'], row['identity']))
    return {'schema_version': 1,
            'method': 'Recorded search and successful open/click responses; exact title/full-text paragraph identities',
            'limitations': [
                'A successful document response is an exposure proxy, not proof the agent read or used its content.',
                'Any opened chunk counts once toward its paragraph; it does not establish full-paragraph reading.',
                'Search snippets may already suffice to answer; a shown-but-never-opened support is not a semantic failure.',
                'Gold labels are used only for scoring. Wiki and collection indexes are reported separately, excluded from document precision.',
                'Cross-question exact title/text copies are one identity; differing text or title is distinct.',
                'All metrics are assignment-local. Earlier assignments do not count as current reads or search exposures.'],
            'definitions': {
                'opened_document_precision': 'Distinct opened supporting paragraphs / all distinct opened document paragraphs.',
                'opened_support_recall': 'Distinct opened supporting paragraphs / all supporting paragraph identities.',
                'shown_support_then_subsequently_opened': 'Shown supporting identities with any later open / all shown supporting identities.',
                'newly_shown_support_selection': 'First-exposed, previously unopened supports opened later / all first-exposed, previously unopened supports.',
                'already_opened_before_first_exposure': 'Supporting identities with an open audit ID lower than their first search exposure ID.',
                'shown_but_never_opened': 'Shown supporting identities with no successful document open in the assignment.',
                'aggregation': 'Micro totals over assignment-local distinct identities; zero denominators have null rates.'},
            'inputs_sha256': {name: hashlib.sha256(value).hexdigest() for name, value in raw.items()},
            'audit_sha256': hashlib.sha256(json.dumps(audit).encode()).hexdigest(),
            'implementation_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'summary': aggregate(assignments), 'assignments': assignments, 'shown_but_never_opened': skipped}


def wiki_url(url):
    if not isinstance(url, str):
        return False
    parsed = urlsplit(url)
    return parsed.scheme == 'https' and parsed.netloc == 'wiki.test' and not parsed.fragment


def cell(value):
    return str(value).replace('|', '\\|').replace('\n', ' ').replace('\r', ' ')


def format_fraction(value):
    count = f"{value['numerator']}/{value['denominator']}"
    return count + (f" ({value['rate']:.1%})" if value['rate'] is not None else ' (n/a)')


def markdown(report):
    lines = ['# Recorded document selection', '', *report['limitations'], '', '| Metric | Result |', '| --- | ---: |']
    summary = report['summary']
    for name in ('opened_document_precision', 'opened_support_recall', 'shown_support_then_subsequently_opened',
                 'newly_shown_support_selection', 'all_support_opened'):
        lines.append(f"| {name.replace('_', ' ')} | {format_fraction(summary[name])} |")
    for name in ('assignment_count', 'zero_search_assignments', 'zero_document_open_assignments',
                 'already_opened_before_first_exposure', 'already_opened_and_reopened_after_exposure',
                 'shown_but_never_opened', 'document_open_calls', 'wiki_open_calls', 'index_open_calls', 'failed_calls'):
        lines.append(f"| {name.replace('_', ' ')} | {summary[name]} |")
    lines += ['', 'Fractions sum assignment-local distinct identities. Wiki/index counts are successful page responses; '
              'repeated responses count again. Saves and errors are not reads.', '',
              'Newly shown support selection excludes supports already opened before first search exposure. '
              'Shown support then subsequently opened includes later reopens of those supports.', '',
              '| Slot | Agent | Question ID | Document precision | Support recall | New support selection | Already opened | Never opened |',
              '| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for a in report['assignments']:
        m = a['metrics']
        lines.append(f"| {a['slot']} | {cell(a['agent'])} | {cell(a['id'])} | "
                     f"{format_fraction(m['opened_document_precision'])} | {format_fraction(m['opened_support_recall'])} | "
                     f"{format_fraction(m['newly_shown_support_selection'])} | "
                     f"{m['already_opened_before_first_exposure']} | {m['shown_but_never_opened']} |")
    lines += ['', '## Shown supporting paragraphs with no recorded document open', '',
              'These are selection observations; the search snippet may have supplied the needed information.', '',
              '| First exposure audit ID | Agent | Slot | Question | Supporting title |',
              '| ---: | --- | ---: | --- | --- |']
    for row in report['shown_but_never_opened']:
        lines.append(f"| {row['first_exposure_audit_id']} | {cell(row['agent'])} | {row['slot']} | "
                     f"{cell(row['question'])} | {cell(row['title'])} |")
    return '\n'.join(lines) + '\n'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args(argv)
    output = args.output_dir or args.run_root / 'selection-audit'
    try:
        require(not output.exists() and not output.is_symlink(), f'Output directory already exists: {output}')
        report = audit_selection(args.run_root)
        output.mkdir(parents=True, exist_ok=False)
        (output / 'report.json').write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + '\n')
        (output / 'report.md').write_text(markdown(report))
    except (OSError, ValueError, KeyError, TypeError, sqlite3.Error) as error:
        parser.exit(2, f'Selection audit failed: {error}\n')
    print(f'Selection report: {output / "report.md"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
