import pytest

import os
import numpy as np

from typing import List

from svs.kb import (
    _DB,
    SQLITE_IS_STRICT,
)


# Things to test:
# - transactions (commit, rollback)
# - unique keys
# - foreign keys


_DB_PATH = './testdb.sqlite'


@pytest.fixture(autouse=True)
def clear_database():
    if os.path.exists(_DB_PATH):
        os.unlink(_DB_PATH)
    assert not os.path.exists(_DB_PATH)
    yield


def test_keyval_table():
    # Open the database
    db = _DB(_DB_PATH)
    with db as q:
        assert q._debug_keyval() == {}
    with db as q:
        for key in ['DNE', 'a', 'b']:
            with pytest.raises(KeyError):
                q.get_key(key)
    with db as q:
        q.set_key('a', 77)
        assert q.get_key('a') == 77
    with db as q:
        q.set_key('b', '99')
        # This is why *strict* mode is better:
        if SQLITE_IS_STRICT:
            assert q.get_key('b') == '99'
        else:
            assert q.get_key('b') == 99
        q.set_key('b', 'hi')
    with db as q:
        assert q._debug_keyval() == {
            'a': 77,
            'b': 'hi',
        }
    db.close()

    # **Re-open** the database
    db = _DB(_DB_PATH)
    with db as q:
        assert q.get_key('a') == 77
        assert q.get_key('b') == 'hi'
    with db as q:
        for key in ['DNE']:
            with pytest.raises(KeyError):
                q.get_key(key)
    with db as q:
        assert q._debug_keyval() == {
            'a': 77,
            'b': 'hi',
        }
    db.close()

    # **Re-open** the database
    db = _DB(_DB_PATH)
    with db as q:
        assert q.get_key('a') == 77
        assert q.get_key('b') == 'hi'
    with db as q:
        for key in ['DNE']:
            with pytest.raises(KeyError):
                q.get_key(key)
    with db as q:
        with pytest.raises(KeyError):
            q.del_key('c')
        q.del_key('b')
    with db as q:
        for key in ['DNE', 'b', 'c']:
            with pytest.raises(KeyError):
                q.get_key(key)
    with db as q:
        assert q._debug_keyval() == {
            'a': 77,
        }
    db.close()

    # **Re-open** the database
    db = _DB(_DB_PATH)
    with db as q:
        assert q.get_key('a') == 77
    with db as q:
        for key in ['DNE', 'b']:
            with pytest.raises(KeyError):
                q.get_key(key)
    with db as q:
        q.set_key('a', 'new val')
        assert q.get_key('a') == 'new val'
    with db as q:
        assert q.get_key('a') == 'new val'
    with db as q:
        assert q._debug_keyval() == {
            'a': 'new val',
        }
    db.close()

    # **Re-open** the database
    db = _DB(_DB_PATH)
    with db as q:
        assert q.get_key('a') == 'new val'
    with db as q:
        for key in ['DNE', 'b']:
            with pytest.raises(KeyError):
                q.get_key(key)
        q.set_key('b', b'buffer val')
    with db as q:
        assert q._debug_keyval() == {
            'a': 'new val',
            'b': b'buffer val',
        }
    db.close()


def test_doc_table():
    # Open the database
    db = _DB(_DB_PATH)
    with db as q:
        doc_a_id = q.add_doc(
            text      = 'first doc',
            parent_id = None,
            meta      = None,
            embedding = b'\x00\x00\x80?',
        )
        assert doc_a_id == 1
    db.close()

    # **Re-open** the database
    db = _DB(_DB_PATH)
    with db as q:
        doc_a = q.fetch_doc(1)
        assert doc_a == {
            'id': 1,
            'parent_id': None,
            'level': 0,
            'text': 'first doc',
            'embedding': [1.0],
            'meta': None,
        }
    db.close()
