"""Three separate, opt-in removals from the frozen 12a reference contract."""
import copy

ARMS={'generic-titles-v1':'12b','own-only-notebooks-v1':'12c','union-discovery-v1':'12d'}


def configure(settings,policy):
    if policy is None:return
    if policy not in ARMS or settings.get('continuation_policy')!='12a-v1':raise ValueError('Invalid reference ablation')
    if settings.get('notebook_title_policy')!='agent-first-line-v1' or settings.get('agent_log_policy')!='no-agent-history-v1':raise ValueError('Ablations require frozen title instructions and no history')
    settings['ablation_policy']=policy
    if policy=='generic-titles-v1':settings['notebook_title_render_policy']='generic-provenance-v1'
    elif policy=='own-only-notebooks-v1':settings['notebook_visibility_policy']='own-only-v1'
    else:
        plan=copy.deepcopy(settings['access_plan'])
        settings['original_access_plan']=copy.deepcopy(plan)
        groups=sorted(set().union(*(set(v) for v in plan['groups'].values())))
        urls=sorted(set().union(*(set(v) for v in plan['allowed_urls'].values())))
        plan.update(groups={a:groups for a in plan['groups']},allowed_urls={a:urls for a in plan['allowed_urls']},overlap_groups=groups)
        settings['access_plan']=plan
