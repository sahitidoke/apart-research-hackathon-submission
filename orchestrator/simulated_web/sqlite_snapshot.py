"""Restore already-verified SQLite bytes without optional serialize APIs."""
from contextlib import closing, contextmanager
from pathlib import Path
import sqlite3
import tempfile


@contextmanager
def snapshot_connection(snapshot):
    """Yield a private writable clone; never reopen the original snapshot path.

    Some Python SQLite builds omit deserialize. The standard backup API works
    from a private immutable read-only file, and permits legacy clone migrations.
    Temporary files and connections are cleaned up on success and failure.
    """
    if not isinstance(snapshot, bytes):
        raise TypeError('Snapshot must be already-verified bytes')
    with closing(sqlite3.connect(':memory:')) as clone:
        with tempfile.TemporaryDirectory(prefix='agent-swarming-snapshot-') as directory:
            path = Path(directory)/'snapshot.sqlite3'
            path.write_bytes(snapshot)
            with closing(sqlite3.connect(path.as_uri()+'?mode=ro&immutable=1', uri=True)) as source:
                source.backup(clone)
        yield clone
