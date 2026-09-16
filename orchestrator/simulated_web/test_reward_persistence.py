"""Stated incentive, persistence and host-only search counts; no grading or models."""
import json
from pathlib import Path
import tempfile
import unittest

from orchestrator.simulated_web.runner import ModelResponse
from orchestrator.simulated_web.test_crossed_access import CrossedAccessTests
from orchestrator.simulated_web.test_token_pair import Client
from orchestrator.simulated_web.token_pair import PERSISTENCE_PROMPT, REWARD_PROMPT, load_pair_checkpoint, pair_policy, pair_session_prompt, run_token_pair


class SearchClient(Client):
    def __call__(self, agent, history, timeout, **kwargs):
        self.calls.append((agent,json.loads(json.dumps(history))))
        if history[-1]['role']=='user':
            calls=[{'function':{'name':'search','arguments':{'query':'evidence'}}},
                   {'function':{'name':'search','arguments':{'query':123}}},
                   {'function':{'name':'open','arguments':{'url':'https://docs.test/'}}}]
            return ModelResponse({'content':'','tool_calls':calls},{'eval_count':3,'prompt_eval_count':100,'done_reason':'stop'})
        return ModelResponse({'content':'mock findings'},{'eval_count':3,'prompt_eval_count':100,'done_reason':'stop'})


class RewardTests(unittest.TestCase):
    def test_prompt_and_search_accounting_checkpoint(self):
        records,selectors,manifest,ids=CrossedAccessTests().setup_inputs()
        policy=pair_policy(compaction_enabled=False,preparation_browser_calls=4,answer_browser_calls=4)
        baseline=pair_session_prompt(policy,'question_research')
        reward=pair_session_prompt(policy,'question_research','reward_persistence')
        self.assertEqual(reward,baseline+'\n\n'+PERSISTENCE_PROMPT+'\n\n'+REWARD_PROMPT)
        self.assertNotIn('1 - 0.1',baseline)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'run';client=SearchClient()
            run_token_pair(path,records,'topic',client,policy,selectors,pair_protocol='question_research',
                           question_ids=ids,access_manifest=manifest,log_exposure='forced',prompt_condition='reward_persistence')
            self.assertEqual(client.calls[0][1][0]['content'],reward)
            self.assertNotIn('PRIVATE_GOLD',reward)
            rows=json.loads((path/'results.json').read_text())
            self.assertEqual(len(rows),12)
            self.assertTrue(all(r['search_calls']==2 for r in rows))
            answers=[r for r in rows if r['phase']=='answer']
            self.assertEqual(len(answers),6)
            for r in answers:
                self.assertEqual(r['stated_reward_accounting'],{'search_calls_research_and_answer':4,
                    'conditional_reward_if_correct':0.95,'correctness_evaluated':False,'reward_evaluated':False})
            cp=load_pair_checkpoint(path/'checkpoints/rounds-003')
            try:
                self.assertEqual(cp.data['settings.json']['prompt_condition'],'reward_persistence')
                self.assertFalse(cp.data['settings.json']['stated_reward']['reward_evaluated'])
                self.assertEqual(cp.browser.db.execute("SELECT count(*) FROM audit WHERE operation='search'").fetchone()[0],24)
            finally:cp.close()
            with self.assertRaises(ValueError):
                run_token_pair(Path(tmp)/'bad',None,None,Client(),resume_from=path/'checkpoints/rounds-003',prompt_condition='baseline')
            self.assertFalse((Path(tmp)/'bad').exists())
            self.assertEqual(run_token_pair(Path(tmp)/'done',None,None,Client(),resume_from=path/'checkpoints/rounds-003')['status'],'already_complete')

    def test_invalid_condition_before_output(self):
        records,selectors,manifest,ids=CrossedAccessTests().setup_inputs()
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'bad'
            with self.assertRaises(ValueError):
                run_token_pair(path,records,'topic',Client(),pair_policy(),selectors,prompt_condition='reward_persistence')
            self.assertFalse(path.exists())
