"""Mock the documented CLI selection contract; no network or artifact downloads."""
import argparse
import ast
from pathlib import Path
import unittest

from orchestrator.simulated_web.hf_fp8 import MODEL, REVISION, artifact_lock, download_command


def parse_selection(arguments):
    parser = argparse.ArgumentParser()
    parser.add_argument('repo_id')
    parser.add_argument('filenames', nargs='*')
    parser.add_argument('--revision')
    parser.add_argument('--cache-dir')
    parser.add_argument('--include', action='append')
    # HF's pinned Typer CLI uses variadic filenames and repeatable single-value include.
    return parser.parse_intermixed_args(arguments)


class DownloadCommandTests(unittest.TestCase):
    def test_all_locked_files_are_explicit_and_cache_is_reused(self):
        expected = [row['rfilename'] for row in artifact_lock()['files']]
        command = download_command('/hf')
        selection = parse_selection(command[2:])
        self.assertEqual(command[:2], ['hf', 'download'])
        self.assertEqual(selection.repo_id, MODEL)
        self.assertEqual(selection.filenames, expected)
        self.assertEqual(len(selection.filenames), 76)
        self.assertEqual(selection.filenames[0], 'chat_template.jinja')
        self.assertEqual(selection.revision, REVISION)
        self.assertEqual(selection.cache_dir, '/hf')
        self.assertIsNone(selection.include)
        self.assertNotIn('--force-download', command)
        # Reproduce the old omission using the same option arity/selection behavior.
        old = parse_selection([MODEL, '--include', *expected, '--revision', REVISION, '--cache-dir', '/hf'])
        self.assertEqual(old.include, ['chat_template.jinja'])
        self.assertEqual(old.filenames, expected[1:])

    def test_all_three_entrypoints_call_shared_command(self):
        root = Path(__file__).parent
        for name in ('modal_hf_paired_views.py', 'modal_async_notebooks.py', 'modal_parallel_notebooks.py'):
            tree = ast.parse((root / name).read_text())
            calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                     and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
                     and node.func.value.id == 'subprocess' and node.func.attr == 'run']
            self.assertEqual(len(calls), 1)
            command = calls[0].args[0]
            self.assertIsInstance(command, ast.Call)
            self.assertEqual(command.func.id, 'download_command')
            self.assertEqual(ast.literal_eval(command.args[0]), '/hf')
