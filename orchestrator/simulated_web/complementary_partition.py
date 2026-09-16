"""Host-only engineered exact-paragraph split; never supplied to browser pages."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

from orchestrator.simulated_web.source_discovery import dataset_digest

VERSION = 'complementary-exact-groups-11b-v1'
SCHEMA = 'source-access-groups-v1'
AGENTS = ('agent-1', 'agent-2')


def identity(paragraph):
    return hashlib.sha256((paragraph['title'] + '\0' + paragraph['paragraph_text']).encode()).hexdigest()


def build_partition(records, question_ids, seed=11):
    """Enumerate annotated support assignments; greedily balance unique source bytes."""
    if type(seed) is not int or not isinstance(records, list) or not records:
        raise ValueError('Nonempty records and integer seed required')
    if (not isinstance(question_ids, dict) or set(question_ids) != set(AGENTS)
            or question_ids['agent-1'] != question_ids['agent-2']
            or len(question_ids['agent-1']) != 2 or len(set(question_ids['agent-1'])) != 2):
        raise ValueError('Exactly two identical unique selected questions required')
    groups, by_id = {}, {}
    for record in records:
        qid = record['id']
        if qid in by_id:
            raise ValueError('Duplicate question ID')
        by_id[qid] = record
        seen = set()
        for position, p in enumerate(record['paragraphs']):
            if p['idx'] in seen or not isinstance(p['title'], str) or not isinstance(p['paragraph_text'], str):
                raise ValueError('Invalid paragraph identity or text')
            seen.add(p['idx'])
            group = groups.setdefault(identity(p), {'title': p['title'], 'source_bytes': len((p['title'] + p['paragraph_text']).encode()), 'occurrences': []})
            group['occurrences'].append({'question_id': qid, 'position': position, 'idx': p['idx']})
    support, terminal = {}, {}
    for qid in question_ids['agent-1']:
        if qid not in by_id:
            raise ValueError('Unknown selected question')
        record = by_id[qid]
        paragraphs = {p['idx']: p for p in record['paragraphs']}
        hops = record.get('question_decomposition', [])
        if not hops or any(h.get('paragraph_support_idx') not in paragraphs for h in hops):
            raise ValueError('Missing decomposition support annotation')
        support[qid] = sorted({identity(p) for p in record['paragraphs'] if p.get('is_supporting') is True})
        hop_groups = {identity(paragraphs[h['paragraph_support_idx']]) for h in hops}
        if hop_groups != set(support[qid]) or len(hop_groups) < 2:
            raise ValueError('Supporting flags/decomposition disagree or split is infeasible')
        terminal[qid] = identity(paragraphs[hops[-1]['paragraph_support_idx']])
    exact_support = sorted(set().union(*map(set, support.values())))
    # Conservative document families: same title OR identical body must co-locate.
    parent = {g: g for g in groups}
    def root(g):
        while parent[g] != g:
            g = parent[g]
        return g
    seen_titles, seen_bodies = {}, {}
    for record in records:
        for paragraph in record['paragraphs']:
            g = identity(paragraph)
            for table, key in ((seen_titles, paragraph['title']), (seen_bodies, paragraph['paragraph_text'])):
                if key in table:
                    parent[root(g)] = root(table[key])
                else:
                    table[key] = g
    families = {}
    for g in groups:
        families.setdefault(root(g), []).append(g)
    family_of = {g: min(members) for members in families.values() for g in members}
    families = {min(members): sorted(members) for members in families.values()}
    family_bytes = {f: sum(groups[g]['source_bytes'] for g in members) for f,members in families.items()}
    support_units = {q: sorted({family_of[g] for g in row}) for q,row in support.items()}
    terminal_units = {q: family_of[g] for q,g in terminal.items()}
    keys = sorted(set().union(*map(set, support_units.values())))
    if len(keys) > 20:
        raise ValueError('Exact enumeration limited to 20 selected supporting groups')
    candidates = []
    for bits in itertools.product((0, 1), repeat=len(keys)):
        assignment = dict(zip(keys, bits))
        counts = [sum(assignment[g] == 0 for g in row) for row in support_units.values()]
        if any(n == 0 or n == len(row) for n, row in zip(counts, support_units.values())):
            continue
        objective = (abs(2*sum(assignment[g] == 0 for g in terminal_units.values())-len(terminal_units)),
                     sum(abs(2*n-len(row)) for n, row in zip(counts, support_units.values())),
                     abs(sum(family_bytes[g] * (1 if assignment[g] == 0 else -1) for g in keys)))
        tie = hashlib.sha256((str(seed)+':'+''.join(map(str,bits))).encode()).hexdigest()
        candidates.append((objective, tie, assignment))
    if not candidates:
        raise ValueError('No complementary supporting-group partition exists')
    objective, _, assigned = min(candidates, key=lambda x: (x[0], x[1]))
    totals = [sum(family_bytes[g] for g in assigned if assigned[g] == a) for a in (0,1)]
    distractors = [0, 0]
    for g in sorted(set(families)-set(keys), key=lambda g: (-family_bytes[g], hashlib.sha256(f'{seed}:{g}'.encode()).hexdigest())):
        a = min((0,1), key=lambda a: (totals[a], distractors[a], a))
        assigned[g] = a
        totals[a] += family_bytes[g]
        distractors[a] += 1
    assigned = {g: assigned[family_of[g]] for g in groups}
    owners = {a: sorted(g for g in groups if assigned[g] == i) for i,a in enumerate(AGENTS)}
    audit = {'schema': VERSION, 'seed': seed, 'dataset_sha256': dataset_digest(records), 'selected_questions': question_ids['agent-1'],
             'objective': {'priority': ['terminal-role imbalance', 'sum per-question support-family-count imbalance', 'support-family source-byte imbalance'], 'exact_support_optimum': list(objective), 'support_candidates': len(candidates), 'remaining_assignment': 'descending document-family unique title/body UTF8 bytes; greedy minimum total bytes, then distractor count; seeded SHA256 ties'},
             'source_bytes': dict(zip(AGENTS, totals)), 'distractor_family_counts': dict(zip(AGENTS, distractors)),
             'groups': {g: {**groups[g], 'owner': AGENTS[assigned[g]]} for g in sorted(groups)},
             'duplicate_groups': sorted(g for g,v in groups.items() if len(v['occurrences']) > 1),
             'document_families': families,
             'family_constraint': 'globally co-locate equal exact titles or exact bodies, transitively',
             'shared_selected_support_groups': sorted(g for g in exact_support if sum(g in s for s in support.values()) > 1),
             'questions': {qid: {'support_groups': support[qid], 'support_counts': {a: sum(g in owners[a] for g in support[qid]) for a in AGENTS}, 'terminal_group': terminal[qid], 'terminal_owner': AGENTS[assigned[terminal[qid]]]} for qid in support},
             'limitations': ['Engineered annotation-based discovery complementarity, not proven semantic necessity.', 'Exact title-NUL-body groups preserve provenance; equal titles or equal bodies additionally co-locate globally. Cross-title alternative evidence, paraphrases and model priors remain unaudited.', 'Foreign known-URL GET, automatic request metadata, and source links remain allowed; notebook dependence is not guaranteed.', 'Terminal role is final decomposition support, not a semantic proof of exclusive answer availability.', 'Paragraph text stays intact; existing browser chunking is unchanged.']}
    manifest = {'schema': SCHEMA, 'dataset_sha256': dataset_digest(records), 'source_groups': owners, 'partition_audit': audit}
    return manifest


def validate_group_manifest(manifest, records, groups):
    if set(manifest) != {'schema','dataset_sha256','source_groups','partition_audit'} or manifest['schema'] != SCHEMA or manifest['dataset_sha256'] != dataset_digest(records):
        raise ValueError('Invalid group access manifest')
    audit = manifest['partition_audit']
    # Recompute rather than trusting saved annotations/assignment or audit claims.
    expected = build_partition(records, {a: audit['selected_questions'] for a in AGENTS}, audit['seed'])
    if manifest != expected or set().union(*map(set, manifest['source_groups'].values())) != set(groups):
        raise ValueError('Group partition differs from deterministic audited assignment')
    return manifest['source_groups']


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--question-ids', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=11)
    args = parser.parse_args(argv)
    manifest = build_partition([json.loads(x) for x in args.dataset.read_text().splitlines() if x.strip()], json.loads(args.question_ids.read_text()), args.seed)
    if args.output.exists():
        raise ValueError('Output already exists; preserve artifacts')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        stream.write(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    main()
