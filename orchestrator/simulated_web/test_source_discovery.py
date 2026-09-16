"""Asymmetric discoverability checks, using temporary browsers and mock models only."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.parse import urlencode

from orchestrator.simulated_web import modal_token_pair as launcher

from orchestrator.simulated_web.browser import Browser
from orchestrator.simulated_web.source_discovery import browser_discovery, discovery_plan
from orchestrator.simulated_web.test_token_pair import Client, inputs
from orchestrator.simulated_web.token_pair import load_pair_checkpoint, pair_policy, run_token_pair, validate_pair


class DiscoveryTests(unittest.TestCase):
    def setup_inputs(self):
        records, selectors = inputs()
        # Duplicate long paragraphs exercise all aliases and chunks without gold labels.
        extra = {'title': 'uniquelyhiddenlong', 'paragraph_text': 'longbody ' * 1000}
        records[0]['paragraphs'].append(extra.copy())
        records[1]['paragraphs'].append(extra.copy())
        policy = pair_policy(compaction_enabled=False)
        pages, tasks, _, editable = validate_pair(records, 'topic', policy, selectors)
        plan = discovery_plan(records, pages, editable, 'asymmetric', 0)
        return records, selectors, policy, pages, tasks, editable, plan

    def test_visibility_direct_urls_history_and_full_equivalence(self):
        records, selectors, policy, pages, tasks, editable, plan = self.setup_inputs()
        browsers = [Browser(pages, ':memory:', editable_sources=editable, request_history_mode='shared',
                            **options) for options in ({}, browser_discovery(plan), browser_discovery(discovery_plan(records, pages, editable)))]
        baseline, asymmetric, full = browsers
        try:
            hidden = asymmetric.discovery_hidden_urls['agent-2']
            self.assertTrue(hidden)
            self.assertIn('search available documents.', asymmetric.open('agent-2', 'https://docs.test/')['text'])
            self.assertIn('search across all documents.', asymmetric.open('agent-1', 'https://docs.test/')['text'])
            self.assertFalse(hidden & asymmetric.editable_urls)
            self.assertEqual(len(plan['shared_groups']), 5)
            readonly = set(plan['groups']) - set(plan['shared_groups'])
            self.assertEqual(len(set(plan['agent_2_visible_groups']) & readonly), (len(readonly) + 1) // 2)
            for urls in plan['groups'].values():
                self.assertIn(len(set(urls) & hidden), (0, len(urls)))
            for query in ['longbody', 'Document', 'uniquelyhiddenlong']:
                self.assertEqual(baseline.search(query), full.search(query, 'agent-2'))
                self.assertEqual(baseline.search(query), asymmetric.search(query, 'agent-1'))
                self.assertFalse(hidden & {r['url'] for r in asymmetric.search(query, 'agent-2')['results']})
            for url in plan['listing_urls']:
                shown = asymmetric.open('agent-2', url)
                self.assertFalse(hidden & {link['url'] for link in shown['links']})
                source = asymmetric.pages[url]
                if source['text'].startswith('Available documents:\n'):
                    expected = [link['label'] for link in source['links'] if link['url'] not in hidden
                                and link['url'] not in plan['listing_urls'] and link['url'].startswith('https://docs.test/')]
                    self.assertEqual(shown['text'], 'Available documents:\n' + '\n'.join(expected))
            target = sorted(hidden)[0]
            result = asymmetric.call('agent-1', 'open', {'url': target})
            self.assertNotIn('error', result)
            log = asymmetric.call('agent-2', 'open', {'url': 'https://docs.test/request-history'})
            self.assertIn(target, log['text'])
            self.assertNotIn('error', asymmetric.call('agent-2', 'open', {'url': target}))
            self.assertEqual(asymmetric.discovery_hidden_urls['agent-2'], hidden)
            identity = next(iter(asymmetric.source_urls))
            saved = asymmetric.call('agent-1', 'open', {'url': 'https://docs.test/source/save?' + urlencode({'source': identity, 'title': 'sharedmarker', 'text': 'sharedvalue'})})
            self.assertIn('saved', saved)
            self.assertEqual(asymmetric.open('agent-2', saved['saved'])['text'], 'sharedvalue')
            self.assertNotEqual(full.open('agent-2', saved['saved'])['text'], 'sharedvalue')
            self.assertEqual(full.db.execute('SELECT count(*) FROM request_events').fetchone()[0], 0)
        finally:
            for browser in browsers:
                browser.close()

    def test_checkpoint_resume_historical_and_tampered_assignment(self):
        records, selectors, policy, _, _, _, _ = self.setup_inputs()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_token_pair(root / 'a', records, 'topic', Client(), policy, selectors, source_discovery='asymmetric')
            path = root / 'a/checkpoints/rounds-005'
            loaded = load_pair_checkpoint(path)
            hidden = loaded.browser.discovery_hidden_urls
            loaded.close()
            run_token_pair(root / 'b', None, None, Client(), resume_from=path)
            loaded = load_pair_checkpoint(root / 'b/checkpoints/rounds-010')
            self.assertEqual(hidden, loaded.browser.discovery_hidden_urls)
            loaded.close()
            with self.assertRaisesRegex(ValueError, 'overrides prohibited'):
                run_token_pair(root / 'invalid', None, None, Client(), resume_from=path, source_discovery='full')
            self.assertFalse((root / 'invalid').exists())
            # A checksum-consistent but recomputation-inconsistent host assignment is rejected.
            settings_path = path / 'settings.json'
            settings = json.loads(settings_path.read_text())
            settings['discovery_plan']['agent_2_visible_groups'] = []
            settings_path.write_text(json.dumps(settings))
            manifest_path = path / 'checkpoint.json'
            manifest = json.loads(manifest_path.read_text())
            manifest['files_sha256']['settings.json'] = hashlib.sha256(settings_path.read_bytes()).hexdigest()
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'discovery mismatch'):
                load_pair_checkpoint(path)
            run_token_pair(root / 'full', records, 'topic', Client(), policy, selectors)
            path = root / 'full/checkpoints/rounds-005'
            settings_path, manifest_path = path / 'settings.json', path / 'checkpoint.json'
            settings = json.loads(settings_path.read_text())
            del settings['source_discovery'], settings['discovery_plan']
            settings_path.write_text(json.dumps(settings))
            manifest = json.loads(manifest_path.read_text())
            manifest['files_sha256']['settings.json'] = hashlib.sha256(settings_path.read_bytes()).hexdigest()
            manifest_path.write_text(json.dumps(manifest))
            loaded = load_pair_checkpoint(path)
            self.assertEqual(loaded.browser.discovery_hidden_urls, {'agent-2': set()})
            loaded.close()

    def test_cli_condition_and_resume_override(self):
        records, selectors = inputs()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'dataset').write_text('\n'.join(json.dumps(r) for r in records))
            (root / 'topic').write_text('topic')
            (root / 'selectors').write_text(json.dumps(selectors))
            (root / 'modal.toml').write_text('[research-profile]\n')
            args = ['--run-id', 'test', '--dataset', str(root / 'dataset'), '--topic-file', str(root / 'topic'), '--editable-sources', str(root / 'selectors')]
            with patch.dict('os.environ', {'MODAL_PROFILE': 'research-profile', 'MODAL_CONFIG_PATH': str(root / 'modal.toml')}), patch.object(launcher.app, 'run') as cloud, patch('builtins.print') as output:
                launcher.main(args + ['--source-discovery', 'asymmetric'])
                report = json.loads(output.call_args.args[0])
                self.assertEqual(report['source_discovery']['mode'], 'asymmetric')
                cloud.assert_not_called()
                with self.assertRaisesRegex(ValueError, 'overrides prohibited'):
                    launcher.main(['--run-id', 'child', '--resume-from', 'parent/checkpoints/rounds-005', '--source-discovery', 'full'])

    def test_hidden_index_title_is_not_searchable(self):
        pages = [
            {'url': 'https://docs.test/p/0/0', 'title': 'secretneedle', 'text': 'private source'},
            {'url': 'https://docs.test/p/1/0', 'title': 'public', 'text': 'public source'},
            {'url': 'https://docs.test/', 'title': 'Document collection', 'text': 'Available documents:\nsecretneedle\npublic',
             'links': [{'label': 'secretneedle', 'url': 'https://docs.test/p/0/0'}, {'label': 'public', 'url': 'https://docs.test/p/1/0'}]}]
        browser = Browser(pages, ':memory:', discovery_hidden_urls={'agent-2': ['https://docs.test/p/0/0']}, discovery_listing_urls=['https://docs.test/'])
        try:
            self.assertTrue(browser.search('secretneedle', 'agent-1')['results'])
            self.assertEqual(browser.search('secretneedle', 'agent-2')['results'], [])
            self.assertNotIn('secretneedle', browser.open('agent-2', 'https://docs.test/')['text'])
            self.assertEqual(browser.open('agent-2', 'https://docs.test/p/0/0')['text'], 'private source')
        finally:
            browser.close()
