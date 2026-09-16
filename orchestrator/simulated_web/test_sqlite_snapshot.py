"""Portable snapshot loading on SQLite builds without serialization APIs."""
from pathlib import Path
import sqlite3
import tempfile

import pytest

from orchestrator.simulated_web.sqlite_snapshot import snapshot_connection


class NoDeserialize(sqlite3.Connection):
    @property
    def deserialize(self):
        raise AttributeError('deserialize is unavailable in this build')


@pytest.fixture
def no_deserialize(monkeypatch):
    original=sqlite3.connect
    def connect(*args,**kwargs):
        kwargs['factory']=NoDeserialize
        return original(*args,**kwargs)
    monkeypatch.setattr(sqlite3,'connect',connect)


def test_clone_writable_snapshot_immutable_and_cleanup(tmp_path,monkeypatch,no_deserialize):
    source=tmp_path/'source.sqlite3'
    with sqlite3.connect(source) as db:
        db.execute('CREATE TABLE example(value TEXT)');db.execute('INSERT INTO example VALUES(?)',('original',))
    before=source.read_bytes();directories=[];original=tempfile.TemporaryDirectory
    def directory(**kwargs):
        result=original(dir=tmp_path,**kwargs);directories.append(Path(result.name));return result
    monkeypatch.setattr('orchestrator.simulated_web.sqlite_snapshot.tempfile.TemporaryDirectory',directory)
    with snapshot_connection(before) as clone:
        assert not hasattr(clone,'deserialize')
        assert clone.execute('SELECT value FROM example').fetchone()==('original',)
        clone.execute('CREATE TABLE legacy_upgrade(id INTEGER)')
        clone.execute('UPDATE example SET value=?',('changed',))
        assert all(not p.exists() for p in directories)
    assert source.read_bytes()==before
    with pytest.raises(sqlite3.ProgrammingError):clone.execute('SELECT 1')


def test_malformed_snapshot_cleans_temporary_files(tmp_path,monkeypatch,no_deserialize):
    original=tempfile.TemporaryDirectory;directories=[]
    def directory(**kwargs):
        result=original(dir=tmp_path,**kwargs);directories.append(Path(result.name));return result
    monkeypatch.setattr('orchestrator.simulated_web.sqlite_snapshot.tempfile.TemporaryDirectory',directory)
    with pytest.raises(sqlite3.DatabaseError):
        with snapshot_connection(b'not a sqlite database') as db:db.execute('PRAGMA integrity_check').fetchall()
    assert directories and all(not path.exists() for path in directories)
