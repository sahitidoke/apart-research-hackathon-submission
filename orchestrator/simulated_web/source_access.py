"""Host-only hard source-corpus access plans; browser history remains shared."""
import hashlib

from orchestrator.simulated_web.source_discovery import dataset_digest
from orchestrator.simulated_web.complementary_partition import SCHEMA, validate_group_manifest


def access_plan(records, pages, discovery, manifest=None, shared_wiki=False, source_access_mode="hard"):
    if source_access_mode not in ("hard", "discovery_only"):
        raise ValueError("Invalid source access mode")
    if type(shared_wiki) is not bool:
        raise ValueError("shared_wiki must be boolean")
    if manifest is None:
        return None
    if isinstance(manifest, dict) and manifest.get('schema') == SCHEMA:
        assigned = validate_group_manifest(manifest, records, discovery['groups'])
        allowed = {agent: sorted({url for group in groups for url in discovery['groups'][group]} | set(discovery['listing_urls']))
                   for agent, groups in assigned.items()}
        return {'source_access_mode': source_access_mode, 'schema': 'source-access-plan-v1',
                'allowed_urls': allowed, 'groups': assigned, 'overlap_groups': [],
                'question_collection': 'https://docs.test/', 'wiki_access': 'shared' if shared_wiki else 'blocked',
                'request_history': 'unchanged'}
    if (not isinstance(manifest, dict) or set(manifest) != {'schema', 'dataset_sha256', 'corpus_questions'}
            or manifest['schema'] != 'source-access-v1' or manifest['dataset_sha256'] != dataset_digest(records)):
        raise ValueError('Invalid source access manifest or dataset digest')
    owners = manifest['corpus_questions']
    by_id = {r['id']: r for r in records}
    if not isinstance(owners, dict) or set(owners) != {'agent-1', 'agent-2'}:
        raise ValueError('Access manifest requires both agents')
    allowed, assigned = {}, {}
    for agent, ids in owners.items():
        if (not isinstance(ids, list) or not ids or any(not isinstance(qid, str) or qid not in by_id for qid in ids)
                or len(set(ids)) != len(ids)):
            raise ValueError('Access corpus requires unique known question IDs')
        groups = {hashlib.sha256((p['title']+'\0'+p['paragraph_text']).encode()).hexdigest()
                  for qid in ids for p in by_id[qid]['paragraphs']}
        assigned[agent] = sorted(groups)
        urls = {url for group in groups for url in discovery['groups'][group]}
        prefixes = {'https://docs.test/q/'+hashlib.sha256(qid.encode()).hexdigest()+'/' for qid in ids}
        urls.update(p['url'] for p in pages if p['url'] in discovery['listing_urls'] and any(p['url'].startswith(prefix) for prefix in prefixes))
        allowed[agent] = sorted(urls | {'https://docs.test/'})
    if set(assigned['agent-1']) | set(assigned['agent-2']) != set(discovery['groups']):
        raise ValueError('Hard private corpora must cover the combined corpus')
    return {**({'source_access_mode':source_access_mode} if source_access_mode != 'hard' else {}), 'schema':'source-access-plan-v1', 'allowed_urls':allowed, 'groups':assigned,
            'overlap_groups': sorted(set(assigned['agent-1']) & set(assigned['agent-2'])), 'question_collection':'https://docs.test/', 'wiki_access':'shared' if shared_wiki else 'blocked', 'request_history':'unchanged'}


def access_browser_options(plan):
    return {'access_allowed_urls': plan['allowed_urls'], 'source_access_mode':plan.get('source_access_mode','hard')} if plan is not None else {}
