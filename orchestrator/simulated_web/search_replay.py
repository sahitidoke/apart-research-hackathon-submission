"""Offline counterfactual retrieval on frozen queries and historical wiki revisions."""
import argparse
from bisect import bisect_right
import hashlib
import json
from pathlib import Path
import sqlite3
from urllib.parse import parse_qs, urlsplit

from orchestrator.simulated_web.browser import Browser
from orchestrator.simulated_web.musique_batch import convert

CUTOFFS = (3, 5, 10)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def identity(paragraph):
    return hashlib.sha256(json.dumps([paragraph['title'], paragraph['paragraph_text']],
                                   ensure_ascii=False).encode()).hexdigest()


def support_index(records, pages):
    """Labels are used only in host-side metrics, never in Browser inputs."""
    expected, url_identity, supports, descriptions = [], {}, {}, {}
    for record in records:
        qid = record['id']
        require(isinstance(qid, str) and qid and qid not in supports, 'Invalid or duplicate dataset ID')
        prefix = 'https://docs.test/q/' + hashlib.sha256(qid.encode()).hexdigest() + '/'
        supports[qid] = set()
        for paragraph in record['paragraphs']:
            key = identity(paragraph)
            descriptions[key] = {'title': paragraph['title'], 'text': paragraph['paragraph_text']}
            require(type(paragraph['is_supporting']) is bool, 'Invalid supporting label')
            if paragraph['is_supporting']:
                supports[qid].add(key)
        for page in convert(record):
            old = page['url']
            page['url'] = old.replace('https://docs.test/', prefix, 1)
            for link in page.get('links', []):
                if link['url'].startswith('https://docs.test/'):
                    link['url'] = link['url'].replace('https://docs.test/', prefix, 1)
            expected.append(page)
            if old.startswith('https://docs.test/p/'):
                position = int(old.split('/')[-2])
                url_identity[page['url']] = identity(record['paragraphs'][position])
    require(expected == pages, 'pages.json does not match dataset paragraph namespaces/content')
    return url_identity, supports, descriptions


def comparison(original, replayed, support):
    old, new = set(original), set(replayed)
    return {'original_supports': sorted(old), 'bm25_supports': sorted(new),
            'gained_supports': sorted(new - old), 'lost_supports': sorted(old - new),
            'support_denominator': len(support),
            'original_recall': len(old) / len(support) if support else None,
            'bm25_recall': len(new) / len(support) if support else None,
            'original_any_support': bool(old), 'bm25_any_support': bool(new),
            'original_all_support': bool(support) and old == support,
            'bm25_all_support': bool(support) and new == support}


def aggregate(rows):
    result = {'denominator': len(rows)}
    for k in CUTOFFS:
        values = [row['top_k'][str(k)] for row in rows]
        denominator = sum(v['support_denominator'] for v in values)
        entry = {'support_denominator': denominator,
                 'gained_supports': sum(len(v['gained_supports']) for v in values),
                 'lost_supports': sum(len(v['lost_supports']) for v in values)}
        for method in ('original', 'bm25'):
            hits = sum(len(v[method + '_supports']) for v in values)
            entry[method] = {'support_hits': hits,
                             'micro_recall': hits / denominator if denominator else None}
            for metric in ('any_support', 'all_support'):
                count = sum(v[method + '_' + metric] for v in values)
                entry[method][metric] = {'count': count, 'denominator': len(rows),
                                         'rate': count / len(rows) if rows else None}
        result[str(k)] = entry
    return result


def replay(run_root):
    root = Path(run_root).resolve()
    raw = {name: (root / name).read_bytes() for name in ('dataset.jsonl', 'pages.json', 'results.json')}
    records = [json.loads(line) for line in raw['dataset.jsonl'].splitlines() if line.strip()]
    pages, results = json.loads(raw['pages.json']), json.loads(raw['results.json'])
    url_identity, supports, descriptions = support_index(records, pages)
    settings_path = root / 'settings.json'
    if settings_path.exists():
        settings = json.loads(settings_path.read_text())
        for name in ('dataset', 'pages'):
            filename = name + ('.jsonl' if name == 'dataset' else '.json')
            if name + '_sha256' in settings:
                require(settings[name + '_sha256'] == hashlib.sha256(raw[filename]).hexdigest(),
                        f'{filename} fingerprint mismatch')
    # mode=ro prevents accidental creation/writes; query_only is additional protection.
    database = root / 'web' / 'wiki.sqlite3'
    source = sqlite3.connect(database.as_uri() + '?mode=ro', uri=True)
    try:
        source.execute('PRAGMA query_only=ON')
        source.execute('BEGIN')
        audit = source.execute('SELECT id,agent,operation,args,result FROM audit ORDER BY id').fetchall()
        revisions = source.execute('SELECT id,agent,slug,title,body FROM revisions ORDER BY id').fetchall()
    finally:
        source.close()
    require([row[0] for row in audit] == list(range(1, len(audit) + 1)), 'Audit IDs must be contiguous from 1')
    require([row[0] for row in revisions] == list(range(1, len(revisions) + 1)),
            'Revision IDs must be contiguous from 1')
    assignments, seen = [], set()
    for row in results:
        key = (row['slot'], row['agent'])
        require(key not in seen and row['id'] in supports, 'Duplicate or unknown assignment')
        seen.add(key)
        before, after = (row['wiki_boundary'][side] for side in ('before', 'after'))
        require(all(type(b['audit']) is int and type(b['revisions']) is int for b in (before, after)),
                'Noninteger assignment boundary')
        require(0 <= before['audit'] <= after['audit'] <= len(audit), 'Invalid assignment audit boundary')
        assignments.append({'slot': row['slot'], 'agent': row['agent'], 'id': row['id'],
                            'wiki_boundary': row['wiki_boundary'], 'search_count': 0,
                            'successful_search_count': 0, 'top_k': {}})
    assignments.sort(key=lambda row: (row['slot'], row['agent']))
    browser = Browser(pages, ':memory:')
    queries, save_ids, views, revision_index = [], [], {}, 0
    try:
        for audit_id, agent, operation, args_text, result_text in audit:
            result = json.loads(result_text)
            require(isinstance(result, dict), f'Audit {audit_id}: result must be an object')
            if 'saved' in result:
                require(revision_index < len(revisions), f'Audit {audit_id}: save has no revision')
                revision_id, author, slug, title, body = revisions[revision_index]
                require(author == agent and result['saved'] == 'https://wiki.test/page/' + slug,
                        f'Audit {audit_id}: save/revision agent or slug mismatch')
                args = json.loads(args_text)
                if operation == 'open':
                    save_url = args['url']
                elif operation == 'click':
                    links = views.get((agent, args['page_id']), {})
                    save_url = links.get(args['link_id'], '')
                else:
                    raise ValueError(f'Audit {audit_id}: save has invalid operation')
                parsed = urlsplit(save_url)
                fields = parse_qs(parsed.query, keep_blank_values=True, strict_parsing=True)
                require(parsed.scheme == 'https' and parsed.netloc == 'wiki.test' and parsed.path == '/save'
                        and fields == {'slug': [slug], 'title': [title], 'text': [body]},
                        f'Audit {audit_id}: saved request differs from revision')
                browser.db.execute('INSERT OR REPLACE INTO pages VALUES(?,?,?)', (slug, title, body))
                revision_index += 1
                save_ids.append(audit_id)
            if 'page_id' in result:
                views[(agent, result['page_id'])] = {link['id']: link['url'] for link in result['links']}
            if operation != 'search':
                continue
            matches = [a for a in assignments if a['agent'] == agent
                       and a['wiki_boundary']['before']['audit'] < audit_id <= a['wiki_boundary']['after']['audit']]
            require(len(matches) == 1, f'Audit {audit_id}: search must map to exactly one assignment')
            assignment = matches[0]
            assignment['search_count'] += 1
            args = json.loads(args_text)
            replay_result = browser.call(agent, 'search', args)
            require(('error' in result) == ('error' in replay_result),
                    f'Audit {audit_id}: original/replayed search validity differs')
            query = {'audit_id': audit_id, 'agent': agent, 'slot': assignment['slot'],
                     'id': assignment['id'], 'args': args, 'wiki_revisions_visible': revision_index}
            if 'error' in result:
                query.update(original_error=result['error'], bm25_error=replay_result['error'])
                queries.append(query)
                continue
            old_urls = [p['url'] for p in result['results']]
            new_urls = [p['url'] for p in replay_result['results']]
            known_urls = set(browser.pages) | {'https://wiki.test/'} | {
                'https://wiki.test/page/' + r[0] for r in browser.db.execute('SELECT slug FROM pages')}
            require(len(old_urls) <= 10 and len(set(old_urls)) == len(old_urls)
                    and all(url in known_urls for url in old_urls),
                    f'Audit {audit_id}: invalid or historically unavailable original result URL')
            query.update(original_urls=old_urls, bm25_urls=new_urls, top_k={})
            assignment['successful_search_count'] += 1
            support = supports[assignment['id']]
            for k in CUTOFFS:
                hits = [{url_identity[u] for u in urls[:k] if u in url_identity} & support
                        for urls in (old_urls, new_urls)]
                query['top_k'][str(k)] = comparison(*hits, support)
            queries.append(query)
        require(revision_index == len(revisions), 'Revision without a successful audit save')
        for assignment in assignments:
            for side in ('before', 'after'):
                boundary = assignment['wiki_boundary'][side]
                require(boundary['revisions'] == bisect_right(save_ids, boundary['audit']),
                        'Assignment revision boundary disagrees with audit history')
            rows = [q for q in queries if q['agent'] == assignment['agent'] and q['slot'] == assignment['slot']
                    and 'top_k' in q]
            for k in CUTOFFS:
                old, new = (set().union(*(set(q['top_k'][str(k)][method + '_supports']) for q in rows))
                            for method in ('original', 'bm25'))
                assignment['top_k'][str(k)] = comparison(old, new, supports[assignment['id']])
    finally:
        browser.close()
    valid = [q for q in queries if 'top_k' in q]
    return {'schema_version': 1, 'method': 'Current Browser BM25; frozen historical queries and wiki saves',
            'limitations': 'Retrieval-only counterfactual; no regenerated model answers or causal accuracy claim. '
                            'Any chunk counts as a paragraph hit; exact title/text copies share identity. '
                            'Wiki and index results consume ranks but are not supporting paragraph hits.',
            'ranking': {'implementation': 'orchestrator.simulated_web.browser.Browser.search',
                        'browser_sha256': hashlib.sha256(Path(__file__).with_name('browser.py').read_bytes()).hexdigest(),
                        'k1': 1.2, 'b': 0.75, 'cutoffs': list(CUTOFFS)},
            'history_sha256': hashlib.sha256(json.dumps({'audit': audit, 'revisions': revisions},
                                                       sort_keys=True).encode()).hexdigest(),
            'inputs_sha256': {name: hashlib.sha256(value).hexdigest() for name, value in raw.items()},
            'summary': {'search_count': len(queries), 'failed_search_count': len(queries) - len(valid),
                        'zero_search_assignments': sum(a['search_count'] == 0 for a in assignments),
                        'revisions_replayed': revision_index, 'queries': aggregate(valid),
                        'assignments': aggregate(assignments)},
            'supports': descriptions, 'queries': queries, 'assignments': assignments}


def markdown(report):
    lines = ['# Historical search replay', '', report['limitations'], '',
             'Query metrics exclude invalid searches. Assignment metrics include every results.json assignment, '
             'including zero-search assignments. Assignment coverage unions the top-k hits across its queries.', '',
             '| Unit | k | Count | Original any | BM25 any | Original all | BM25 all | Support hits (old/new/total) | Gains | Losses |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |']
    for unit in ('queries', 'assignments'):
        summary = report['summary'][unit]
        for k in CUTOFFS:
            v = summary[str(k)]
            old, new = v['original'], v['bm25']
            lines.append(f"| {unit} | {k} | {summary['denominator']} | {old['any_support']['count']} | "
                         f"{new['any_support']['count']} | {old['all_support']['count']} | {new['all_support']['count']} | "
                         f"{old['support_hits']}/{new['support_hits']}/{v['support_denominator']} | "
                         f"{v['gained_supports']} | {v['lost_supports']} |")
    lines += ['', '| Slot | Agent | Question | Searches | k | Original supports | BM25 supports | Gains | Losses |',
              '| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in report['assignments']:
        for k in CUTOFFS:
            v = row['top_k'][str(k)]
            safe_id = str(row['id']).replace('|', '\\|').replace('\n', ' ')
            lines.append(f"| {row['slot']} | {row['agent']} | {safe_id} | {row['search_count']} | {k} | "
                         f"{len(v['original_supports'])}/{v['support_denominator']} | "
                         f"{len(v['bm25_supports'])}/{v['support_denominator']} | "
                         f"{len(v['gained_supports'])} | {len(v['lost_supports'])} |")
    return '\n'.join(lines) + '\n'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args(argv)
    output = args.output_dir or args.run_root / 'search-replay'
    try:
        require(not output.exists() and not output.is_symlink(), f'Output directory already exists: {output}')
        report = replay(args.run_root)
        output.mkdir(parents=True, exist_ok=False)
        (output / 'report.json').write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + '\n')
        (output / 'report.md').write_text(markdown(report))
    except (OSError, ValueError, KeyError, TypeError, sqlite3.Error) as error:
        parser.exit(2, f'Search replay failed: {error}\n')
    print(f'Replay report: {output / "report.md"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
