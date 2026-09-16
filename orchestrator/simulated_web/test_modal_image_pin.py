"""All HF entrypoints share the verified release tag, without changing package version."""
import unittest
from unittest.mock import patch

from modal.image import _Image

from orchestrator.simulated_web.hf_fp8 import VLLM_VERSION
from orchestrator.simulated_web import modal_hf_paired_views, modal_async_notebooks, modal_parallel_notebooks


class ImagePinTests(unittest.TestCase):
    def test_all_entrypoints_use_official_release_tag(self):
        self.assertEqual(VLLM_VERSION, '0.24.0')
        self.assertEqual(modal_hf_paired_views.IMAGE, 'vllm/vllm-openai:v0.24.0')
        for module in (modal_hf_paired_views, modal_async_notebooks, modal_parallel_notebooks):
            self.assertIs(module.image, modal_hf_paired_views.image)
            self.assertEqual(module.resources()['backend_image'], 'vllm/vllm-openai:v0.24.0')

    def test_existing_system_python_is_bound_before_modal_pip_setup(self):
        module = modal_hf_paired_views
        with patch.object(module.modal.Image, 'from_registry') as factory:
            module.build_image()
            factory.assert_called_once_with(module.IMAGE, setup_dockerfile_commands=module.PYTHON_SETUP)
            base = factory.return_value.entrypoint.return_value
            base.pip_install.assert_called_once_with('transformers==5.8.0')
            base.pip_install.return_value.run_commands.assert_called_once_with(module.VERIFY_ENVIRONMENT)
        commands = _Image._registry_setup_commands(module.IMAGE, '2024.10', module.PYTHON_SETUP)
        alias_index = commands.index(module.PYTHON_SETUP[0])
        pip_index = next(i for i, command in enumerate(commands) if command.startswith('RUN python -m pip'))
        self.assertLess(alias_index, pip_index)
        self.assertIn('/usr/bin/python3 /usr/local/bin/python', commands[alias_index])
        self.assertFalse(any('COPY /python/' in command for command in commands))
