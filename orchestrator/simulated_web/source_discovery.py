"""Host-only source listing assignment; this never restricts direct URL opens."""
import hashlib
import json
import random

from orchestrator.simulated_web.browser import MAX_TEXT


def dataset_digest(records):
    return hashlib.sha256(json.dumps(records, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def validate_evidence_manifest(manifest, records, groups, shared):
    """Validate an explicitly audited host manifest, never infer necessity from labels."""
    if (not isinstance(manifest, dict) or set(manifest) != {'schema', 'dataset_sha256', 'questions'}
            or manifest['schema'] != 'source-discovery-evidence-v1'
            or manifest['dataset_sha256'] != dataset_digest(records)):
        raise ValueError('Invalid evidence manifest schema or dataset digest')
    questions = manifest['questions']
    if not isinstance(questions, dict) or set(questions) != {record['id'] for record in records}:
        raise ValueError('Evidence manifest must cover exactly all question IDs')
    starts, hidden = set(), set()
    for qid, row in questions.items():
        if not isinstance(row, dict) or set(row) != {'starting_groups', 'withheld_groups', 'rationale', 'evidence_note'}:
            raise ValueError('Invalid question evidence fields')
        for field in ('rationale', 'evidence_note'):
            if not isinstance(row[field], str) or not row[field].strip():
                raise ValueError('Evidence manifest needs explicit rationale and evidence note')
        for field, target in (('starting_groups', starts), ('withheld_groups', hidden)):
            identities = row[field]
            if (not isinstance(identities, list) or not identities
                    or any(not isinstance(identity, str) or identity not in groups for identity in identities)
                    or len(set(identities)) != len(identities)):
                raise ValueError('Evidence groups must be nonempty unique known canonical identities')
            target.update(identities)
    if starts & hidden:
        raise ValueError('Withheld evidence conflicts with starting evidence across questions')
    if hidden & shared:
        raise ValueError('Withheld evidence conflicts with shared editable sources')
    return hidden


def discovery_plan(records, pages, editable, mode='full', seed=0, evidence_manifest=None):
    if mode not in ('full', 'asymmetric', 'evidence'):
        raise ValueError('Invalid source discovery mode')
    groups = {}
    shared = set()
    editable = set(editable)
    for record in records:
        prefix = 'https://docs.test/q/' + hashlib.sha256(record['id'].encode()).hexdigest() + '/'
        for position, paragraph in enumerate(record['paragraphs']):
            title, body = paragraph['title'], paragraph['paragraph_text']
            identity = hashlib.sha256((title + '\0' + body).encode()).hexdigest()
            chunks = [body[i:i + MAX_TEXT] for i in range(0, len(body), MAX_TEXT)] or ['']
            urls = groups.setdefault(identity, set())
            for part, chunk in enumerate(chunks):
                urls.add(f'{prefix}p/{position}/{part}')
                if (title[:180] + f' (part {part + 1})', hashlib.sha256(chunk.encode()).hexdigest()) in editable:
                    shared.add(identity)
    if mode != 'evidence' and evidence_manifest is not None:
        raise ValueError('Evidence manifest requires evidence discovery mode')
    readonly = sorted(set(groups) - shared)
    random.Random(seed).shuffle(readonly)
    visible = set(readonly[:(len(readonly) + 1) // 2]) if mode == 'asymmetric' else set(readonly)
    visible |= shared
    if mode == 'evidence':
        visible = set(groups) - validate_evidence_manifest(evidence_manifest, records, groups, shared)
    all_urls = {url for urls in groups.values() for url in urls}
    plan = {'mode': mode, 'seed': seed, 'algorithm': 'sorted original title-NUL-body SHA256; seeded shuffle; agent-2 first ceil(n/2) read-only groups',
            'agent_roles': {'agent-1': 'all', 'agent-2': 'seeded_half' if mode == 'asymmetric' else 'all'},
            'groups': {key: sorted(groups[key]) for key in sorted(groups)},
            'shared_groups': sorted(shared), 'agent_2_visible_groups': sorted(visible),
            'listing_urls': sorted(p['url'] for p in pages if p['url'] not in all_urls)}
    if mode == 'evidence':
        plan['algorithm'] = 'Explicit reviewed host evidence split; union of withheld groups hidden from agent-2 listings'
        plan['agent_roles']['agent-2'] = 'evidence_split'
        plan['evidence_manifest'] = evidence_manifest
    plan['sha256'] = hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return plan


def browser_discovery(plan):
    visible = set(plan['agent_2_visible_groups'])
    hidden = [url for identity, urls in plan['groups'].items() if identity not in visible for url in urls]
    return {'discovery_hidden_urls': {'agent-2': hidden}, 'discovery_listing_urls': plan['listing_urls']}
