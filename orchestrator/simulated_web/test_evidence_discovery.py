"""Host evidence manifest contracts; synthetic assignments are plumbing fixtures."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from orchestrator.simulated_web import modal_token_pair as launcher
from orchestrator.simulated_web.source_discovery import dataset_digest, discovery_plan
from orchestrator.simulated_web.test_token_pair import Client, inputs
from orchestrator.simulated_web.token_pair import load_pair_checkpoint, pair_policy, run_token_pair, validate_pair


def fixture():
    records, selectors = inputs()
    policy = pair_policy(compaction_enabled=False)
    pages, _, _, editable = validate_pair(records, 'topic', policy, selectors)
    full = discovery_plan(records, pages, editable)
    start = full['shared_groups'][0]
    withheld = sorted(set(full['groups']) - set(full['shared_groups']))[0]
    manifest = {'schema': 'source-discovery-evidence-v1', 'dataset_sha256': dataset_digest(records),
                'questions': {r['id']: {'starting_groups': [start], 'withheld_groups': [withheld],
                    'rationale': 'Synthetic plumbing fixture only.', 'evidence_note': 'No claim of task necessity.'} for r in records}}
    return records, selectors, policy, pages, editable, manifest


class EvidenceTests(unittest.TestCase):
    def test_plan_validation_and_no_artifacts_on_invalid(self):
        records, selectors, policy, pages, editable, manifest = fixture()
        plan = discovery_plan(records, pages, editable, 'evidence', evidence_manifest=manifest)
        hidden = set(plan['groups']) - set(plan['agent_2_visible_groups'])
        self.assertEqual(hidden, set(next(iter(manifest['questions'].values()))['withheld_groups']))
        self.assertEqual(plan['evidence_manifest'], manifest)
        self.assertEqual(plan, discovery_plan(records, pages, editable, 'evidence', evidence_manifest=deepcopy(manifest)))
        variants = []
        bad = deepcopy(manifest); bad['dataset_sha256'] = '0' * 64; variants.append(bad)
        bad = deepcopy(manifest); bad['questions'].pop(next(iter(bad['questions']))); variants.append(bad)
        for field, value in [('starting_groups', []), ('withheld_groups', []), ('withheld_groups', ['0' * 64]), ('rationale', ''), ('evidence_note', '')]:
            bad = deepcopy(manifest); next(iter(bad['questions'].values()))[field] = value; variants.append(bad)
        bad = deepcopy(manifest); row = next(iter(bad['questions'].values())); row['starting_groups'] = row['withheld_groups']; variants.append(bad)
        bad = deepcopy(manifest); next(iter(bad['questions'].values()))['withheld_groups'] = [plan['shared_groups'][-1]]; variants.append(bad)
        variants.append(None)
        with tempfile.TemporaryDirectory() as directory:
            for index, bad in enumerate(variants):
                with self.subTest(index=index), self.assertRaises(ValueError):
                    run_token_pair(Path(directory) / str(index), records, 'topic', Client(), policy, selectors,
                                   source_discovery='evidence', evidence_manifest=bad)
                self.assertFalse((Path(directory) / str(index)).exists())
        with self.assertRaises(ValueError):
            discovery_plan(records, pages, editable, 'full', evidence_manifest=manifest)

    def test_resume_preserves_manifest_and_private_labels(self):
        records, selectors, policy, _, _, manifest = fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_token_pair(root / 'a', records, 'topic', Client(), policy, selectors, source_discovery='evidence', evidence_manifest=manifest)
            checkpoint = root / 'a/checkpoints/rounds-005'
            loaded = load_pair_checkpoint(checkpoint)
            hidden = loaded.browser.discovery_hidden_urls
            loaded.close()
            client = Client()
            run_token_pair(root / 'b', None, None, client, resume_from=checkpoint)
            settings = json.loads((root / 'b/settings.json').read_text())
            self.assertEqual(settings['evidence_manifest'], manifest)
            loaded = load_pair_checkpoint(root / 'b/checkpoints/rounds-010')
            self.assertEqual(hidden, loaded.browser.discovery_hidden_urls)
            loaded.close()
            messages = json.dumps(client.calls)
            self.assertNotIn('Synthetic plumbing fixture', messages)
            self.assertNotIn('withheld_groups', messages)
            with self.assertRaises(ValueError):
                run_token_pair(root / 'bad', None, None, Client(), resume_from=checkpoint, evidence_manifest=manifest)

    def test_cli_manifest_validation_without_cloud(self):
        records, selectors, _, _, _, manifest = fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'dataset').write_text('\n'.join(json.dumps(r) for r in records))
            (root / 'topic').write_text('topic')
            (root / 'selectors').write_text(json.dumps(selectors))
            (root / 'manifest').write_text(json.dumps(manifest))
            (root / 'modal.toml').write_text('[research-profile]\n')
            args = ['--run-id', 'test', '--dataset', str(root / 'dataset'), '--topic-file', str(root / 'topic'), '--editable-sources', str(root / 'selectors'), '--source-discovery', 'evidence']
            with patch.dict('os.environ', {'MODAL_PROFILE': 'research-profile', 'MODAL_CONFIG_PATH': str(root / 'modal.toml')}), patch.object(launcher.app, 'run') as cloud, patch('builtins.print') as output:
                with self.assertRaises(ValueError):
                    launcher.main(args)
                launcher.main(args + ['--evidence-manifest', str(root / 'manifest')])
                self.assertEqual(json.loads(output.call_args.args[0])['source_discovery']['evidence_manifest'], manifest)
                with self.assertRaises(ValueError):
                    launcher.main(['--run-id', 'child', '--resume-from', 'parent/checkpoints/rounds-005', '--evidence-manifest', str(root / 'manifest')])
                cloud.assert_not_called()
