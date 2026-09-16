"""Variable question-count schedules and durable checkpoints; mocks only."""
import json
from pathlib import Path
import tempfile
import unittest

from orchestrator.simulated_web.modal_private_notes import resources, validate_selector
from orchestrator.simulated_web.private_notes import build_settings, load_checkpoint, run_private_notes, steps
from orchestrator.simulated_web.test_crossed_access import CrossedAccessTests
from orchestrator.simulated_web.test_mandatory_notes import MandatoryClient


class NotebookCountTests(unittest.TestCase):
    def test_ten_questions_round_four_resume_through_ten(self):
        records,selectors,manifest,_=CrossedAccessTests().setup_inputs()
        ids={a:[r['id'] for r in records] for a in ('agent-1','agent-2')}
        options={'question_count':10,'mandatory_notes':True,'retain_context':True,'sequence_mode':'agent_serial','history_search':True}
        settings,*_=build_settings(records,'topic',selectors,ids,manifest,**options)
        schedule=steps(settings)
        self.assertEqual(len(schedule),60)
        self.assertEqual(schedule[2][0:2],('agent-1','note'))
        self.assertEqual(schedule[3][0:2],('agent-2','research'))
        self.assertEqual(schedule[-1][3],10)
        self.assertTrue(all('10 related questions' in p for p in settings['system_prompts'].values()))
        self.assertEqual(settings['maximum_generated_tokens'],163840)
        self.assertEqual(resources(True,10)['maximum_generated_tokens'],163840)
        with tempfile.TemporaryDirectory() as tmp:
            first=Path(tmp)/'first'
            def stop(cp):
                if cp.name=='rounds-004':raise RuntimeError('mock stop')
            with self.assertRaisesRegex(RuntimeError,'mock stop'):
                run_private_notes(first,MandatoryClient(),records,'topic',selectors,ids,manifest,checkpoint_callback=stop,**options)
            cp=first/'checkpoints/rounds-004'
            data,browser=load_checkpoint(cp);browser.close()
            self.assertEqual(data['state.json']['next_phase_index'],24)
            second=Path(tmp)/'second';client=MandatoryClient()
            result=run_private_notes(second,client,resume_from=cp)
            self.assertEqual(result['status'],'complete')
            final,browser=load_checkpoint(second/'checkpoints/rounds-010');browser.close()
            self.assertEqual(len(final['results.json']),60)
            self.assertEqual(final['state.json']['completed_rounds'],10)
            self.assertEqual(len(client.note_requests),12)
            agent,history=client.calls[0];prior=data['histories.json'][agent]
            self.assertEqual(history[:len(prior)],prior)
            third=Path(tmp)/'third'
            self.assertEqual(run_private_notes(third,MandatoryClient(),resume_from=second/'checkpoints/rounds-010')['status'],'already_complete')
            self.assertFalse(third.exists())

    def test_selector_default_three_new_ten_and_invalid_count_before_output(self):
        validate_selector('parent/checkpoints/rounds-003','child')
        with self.assertRaises(ValueError):validate_selector('parent/checkpoints/rounds-004','child')
        validate_selector('parent/checkpoints/rounds-010','child',10)
        with self.assertRaises(ValueError):validate_selector('parent/checkpoints/rounds-011','child',10)
        records,selectors,manifest,ids=CrossedAccessTests().setup_inputs()
        with tempfile.TemporaryDirectory() as tmp:
            for count in (0,11,True,10):
                with self.subTest(count=count):
                    path=Path(tmp)/str(count)
                    with self.assertRaises(ValueError):
                        run_private_notes(path,MandatoryClient(),records,'topic',selectors,ids,manifest,question_count=count)
                    self.assertFalse(path.exists())
