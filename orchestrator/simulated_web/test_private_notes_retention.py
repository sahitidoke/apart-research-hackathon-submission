"""Full conversation retention regression checks; mock model only."""
import json
from pathlib import Path
import tempfile
import unittest

from orchestrator.simulated_web.private_notes import build_settings, load_checkpoint, run_private_notes
from orchestrator.simulated_web.test_crossed_access import CrossedAccessTests
from orchestrator.simulated_web.test_private_notes import NoteClient


class RetainedNotesTests(unittest.TestCase):
    def test_retained_resume_keeps_conversation_and_browser_results(self):
        records,selectors,manifest,ids=CrossedAccessTests().setup_inputs()
        with tempfile.TemporaryDirectory() as tmp:
            first=Path(tmp)/'first'
            def stop(path):
                if path.name=='rounds-001':raise RuntimeError('mock checkpoint stop')
            client=NoteClient(fail_save=True)
            with self.assertRaisesRegex(RuntimeError,'mock checkpoint stop'):
                run_private_notes(first,client,records,'topic',selectors,ids,manifest,
                    sequence_mode='agent_serial',history_search=True,retain_context=True,checkpoint_callback=stop)
            cp=first/'checkpoints/rounds-001'
            data,browser=load_checkpoint(cp)
            try:
                self.assertTrue(data['settings.json']['retain_context'])
                self.assertEqual(data['settings.json']['policy']['browser_retention'],'full')
                self.assertFalse(data['settings.json']['policy']['compaction_enabled'])
                for agent,history in data['histories.json'].items():
                    self.assertGreater(len(history),2)
                    self.assertIn(agent+' private result',json.dumps(history))
                    self.assertTrue(any(m['role']=='tool' for m in history))
                before=data['histories.json']
            finally:browser.close()
            self.assertFalse(list(first.glob('context-reset-*')))
            for _,history in client.calls:
                for message in history:
                    if message['role'] in ('system','user'):
                        for forbidden in ('cleared','reset','omitted'):
                            self.assertNotIn(forbidden,message['content'].lower())
            resumed=Path(tmp)/'resumed';client2=NoteClient()
            status=run_private_notes(resumed,client2,resume_from=cp)
            self.assertEqual(status['status'],'complete')
            agent,context=client2.calls[0]
            self.assertEqual(context[:len(before[agent])],before[agent])
            self.assertFalse(list(resumed.glob('context-reset-*')))
            data,browser=load_checkpoint(resumed/'checkpoints/rounds-003')
            try:
                for agent,history in data['histories.json'].items():
                    self.assertEqual(history[:len(before[agent])],before[agent])
            finally:browser.close()
            bad=Path(tmp)/'bad'
            with self.assertRaisesRegex(ValueError,'overrides prohibited'):
                run_private_notes(bad,NoteClient(),resume_from=cp,retain_context=False)
            self.assertFalse(bad.exists())

    def test_default_omits_flag_and_rejects_invalid_mode(self):
        records,selectors,manifest,ids=CrossedAccessTests().setup_inputs()
        settings,*_=build_settings(records,'topic',selectors,ids,manifest)
        self.assertNotIn('retain_context',settings)
        self.assertEqual(settings['policy']['browser_retention'],'question_boundary')
        with tempfile.TemporaryDirectory() as tmp:
            bad=Path(tmp)/'bad'
            with self.assertRaisesRegex(ValueError,'boolean'):
                run_private_notes(bad,NoteClient(),records,'topic',selectors,ids,manifest,retain_context='yes')
            self.assertFalse(bad.exists())
