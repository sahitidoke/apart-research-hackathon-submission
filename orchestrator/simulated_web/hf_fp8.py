"""Pinned official HF FP8 artifact contract; importing performs no downloads."""
import hashlib
import json
from pathlib import Path

MODEL = 'Qwen/Qwen3.8-27B-FP8'
REVISION = '017b9c7af6b5689d5dd426a76e0bc077eb5ca20a'
VLLM_VERSION = '0.24.0'
KV_CACHE_DTYPE = 'bfloat16'
LOCK_PATH = Path(__file__).with_name('hf_fp8_lock.json')


def artifact_lock():
    lock = json.loads(LOCK_PATH.read_text())
    if (lock['repo_id'], lock['revision']) != (MODEL, REVISION):
        raise ValueError('HF artifact lock identity mismatch')
    return lock


def download_command(cache_dir):
    """Explicit filenames avoid hf CLI's single-value --include option ambiguity."""
    return ['hf', 'download', MODEL, *[row['rfilename'] for row in artifact_lock()['files']],
            '--revision', REVISION, '--cache-dir', str(cache_dir)]


def validate_snapshot(path):
    """Hash all cached artifacts before setup/run output or GPU initialization."""
    path = Path(path)
    lock = artifact_lock()
    for row in lock['files']:
        file = path / row['rfilename']
        if not file.is_file() or file.stat().st_size != row['size']:
            raise ValueError(f'Missing or wrong-size pinned HF artifact: {file.name}')
    for row in lock['files']:
        file = path / row['rfilename']
        digest = hashlib.sha256() if 'lfs' in row else hashlib.sha1()
        if 'lfs' not in row:
            digest.update(f'blob {row["size"]}\0'.encode())
        with file.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(block)
        expected = row['lfs']['sha256'] if 'lfs' in row else row['blobId']
        if digest.hexdigest() != expected:
            raise ValueError(f'Pinned HF artifact checksum mismatch: {file.name}')
    config = json.loads((path / 'config.json').read_text())
    quant = config.get('quantization_config', {})
    if any(quant.get(k) != v for k, v in {
        'quant_method': 'fp8', 'fmt': 'e4m3', 'activation_scheme': 'dynamic',
        'weight_block_size': [128, 128],
    }.items()):
        raise ValueError('Official fine-grained FP8 configuration required; no fallback')
    return {'name': MODEL, 'revision': REVISION, 'digest': 'hf:' + REVISION,
            'artifact_lock_sha256': hashlib.sha256(LOCK_PATH.read_bytes()).hexdigest(),
            'weight_quantization': 'FP8 E4M3 block128x128, mixed BF16 exclusions',
            'tensor_inventory': lock['safetensors'], 'kv_cache_dtype': KV_CACHE_DTYPE,
            'kv_cache_flag': 'auto with explicit bfloat16 model dtype',
            'runtime_precision_verified': False,
            'backend': 'vllm', 'backend_version': VLLM_VERSION,
            'all_artifact_checksums_verified': True}


def server_command(snapshot, context_length, port, *, max_num_seqs=1):
    if type(context_length) is not int or not 1 <= context_length <= 65536:
        raise ValueError('FP8 pilot context must be within 65536 tokens')
    if type(port) is not int or not 1024 <= port <= 65535:
        raise ValueError('Invalid loopback port')
    if type(max_num_seqs) is not int or max_num_seqs not in (1,2):
        raise ValueError('FP8 pilot supports one or two concurrent sequences')
    return ['vllm', 'serve', str(snapshot), '--served-model-name', MODEL,
            '--host', '127.0.0.1', '--port', str(port), '--quantization', 'fp8',
            '--dtype', 'bfloat16', '--kv-cache-dtype', 'auto',
            '--tensor-parallel-size', '1', '--max-num-seqs', str(max_num_seqs),
            '--max-model-len', str(context_length), '--gpu-memory-utilization', '0.90',
            '--enforce-eager', '--reasoning-parser', 'qwen3',
            '--enable-auto-tool-choice', '--tool-call-parser', 'qwen3_coder',
            '--generation-config', 'vllm']


def validate_client(client):
    if (not callable(client) or getattr(client, 'deadline_cancellation_guaranteed', False) is not True
            or getattr(client, 'native_context_preflight', False) is not True
            or any(not callable(getattr(client, method, None)) for method in
                   ('count_context', 'count_text', 'ensure_ready', 'inspect_model'))):
        raise ValueError('HF pilot requires owned native-context and cancellable transport')
    metadata = client.inspect_model()
    if (metadata.get('name') != MODEL or metadata.get('revision') != REVISION
            or metadata.get('digest') != 'hf:' + REVISION
            or metadata.get('backend') != 'vllm' or metadata.get('backend_version') != VLLM_VERSION
            or metadata.get('kv_cache_dtype') != KV_CACHE_DTYPE
            or metadata.get('all_artifact_checksums_verified') is not True):
        raise ValueError('Pinned HF FP8 model contract mismatch')
    return metadata
