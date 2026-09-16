"""Opt-in distinct question assignment and per-question reset, mock transport only."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from orchestrator.simulated_web.test_token_pair import Client, inputs
from orchestrator.simulated_web.token_pair import load_pair_checkpoint, pair_policy, reset_base_history, run_token_pair


class VariantTests(unittest.TestCase):
    def test_per_agent_selection_combined_corpus_and_isolated_history(self):
        records, selectors = inputs()
        extra = json.loads(json.dumps(records))
        for r in extra:
            r['id'] += '-other'
        combined = records + extra
        ids = {'agent-1': [r['id'] for r in records[:3]], 'agent-2': [r['id'] for r in extra[:3]]}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for mode in ('shared', 'isolated'):
                run = root / mode
                run_token_pair(run, combined, 'topic', Client(), pair_policy(request_history_mode=mode), selectors,
                               pair_protocol='answers_only', question_ids=ids, log_exposure='forced')
                settings = json.loads((run/'settings.json').read_text())
                self.assertEqual(settings['schedule']['orders'], ids)
                self.assertEqual(settings['corpus_question_count'], 20)
                self.assertIn('answer 3 related questions', settings['system_prompt'])
                rows = json.loads((run/'results.json').read_text())
                self.assertEqual([r['question_id'] for r in rows], [ids[a][i] for i in range(3) for a in ids])
                first_b = json.loads((run/'forced-log-01-agent-2.json').read_text())['response']['text']
                self.assertEqual(first_b == 'No requests.', mode == 'isolated')
                cp = load_pair_checkpoint(run/'checkpoints/rounds-003')
                try:
                    self.assertEqual(cp.browser.request_history_mode, mode)
                    self.assertEqual(cp.browser.db.execute('SELECT count(*) FROM request_events').fetchone()[0], 12)
                finally:
                    cp.close()

    def test_reset_after_reflection_and_resume(self):
        records, selectors = inputs()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = Client()
            with patch('orchestrator.simulated_web.token_pair.compact_between_questions') as compact:
                run_token_pair(root/'parent', records, 'topic', client, pair_policy(), selectors, context_reset='after_reflection')
                compact.assert_not_called()
            settings = json.loads((root/'parent/settings.json').read_text())
            base = reset_base_history(settings['system_prompt'], 'topic')
            answers = [(a,h) for a,h in client.calls if h[-1]['role'] == 'user' and 'Question:' in h[-1]['content']]
            self.assertEqual(len(answers), 20)
            for a in ('agent-1','agent-2'):
                own = [h for owner,h in answers if owner == a]
                self.assertGreater(len(own[0]), 3) # preparation retained for Q1
                for history in own[1:]:
                    self.assertEqual(history[:-1], base)
            reflections = [h for _,h in client.calls if h[-1]['role']=='user' and h[-1]['content'].startswith('Reflection and continued research:')]
            self.assertEqual(len(reflections), 20)
            self.assertTrue(all(any('private answer' in m.get('content','') for m in h[:-1]) for h in reflections))
            snapshots = list((root/'parent').glob('context-reset-*'))
            self.assertEqual(len(snapshots), 20)
            self.assertIn('private answer', snapshots[0].read_text())
            cp_path=root/'parent/checkpoints/rounds-005'
            cp=load_pair_checkpoint(cp_path)
            try:
                self.assertEqual(cp.data['histories.json'], {a:base for a in ('agent-1','agent-2')})
                self.assertEqual(cp.browser.db.execute('SELECT count(*) FROM request_events').fetchone()[0],22)
            finally:
                cp.close()
            resumed = Client()
            run_token_pair(root/'child', None, None, resumed, resume_from=cp_path)
            self.assertEqual(resumed.calls[0][1][:-1],base)
            with self.assertRaises(ValueError):
                run_token_pair(root/'bad', None, None, Client(), resume_from=cp_path, context_reset='none')
            self.assertFalse((root/'bad').exists())

    def test_invalid_variants_create_no_artifacts(self):
        records,selectors=inputs()
        ids=[r['id'] for r in records[:3]]
        with tempfile.TemporaryDirectory() as tmp:
            for overrides in ({'pair_protocol':'answers_only','question_ids':ids,'context_reset':'after_reflection'},
                              {'pair_protocol':'answers_only','question_ids':{'agent-1':ids}}, {'context_reset':'bad'}):
                destination=Path(tmp)/'invalid'
                with self.assertRaises(ValueError):
                    run_token_pair(destination, records,'topic',Client(),pair_policy(),selectors,**overrides)
                self.assertFalse(destination.exists())
