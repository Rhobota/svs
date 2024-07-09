import pytest

import os
import json

from svs.embeddings import (
    make_mock_embeddings_func,
    make_openai_embeddings_func,
)

from svs.kb import (
    KB,
    _DB,
    SQLITE_IS_STRICT,
)


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
    with db as q:
        with pytest.raises(KeyError):
            q.del_key('b')
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
            embedding = b'\x00\x00\x80?',  # [1.0]
        )
        assert doc_a_id == 1

        doc_b_id = q.add_doc(
            text      = 'second doc',
            parent_id = 1,
            meta      = None,
            embedding = b'\x00\x00\x00@\x00\x00`@',  # [2.0, 3.5]
        )
        assert doc_b_id == 2

        doc_c_id = q.add_doc(
            text      = 'third doc',
            parent_id = None,
            meta      = {'test': 'stuff'},
            embedding = b'\x00\x00\x00@',   # [2.0]
        )
        assert doc_c_id == 3

        doc_d_id = q.add_doc(
            text      = 'forth doc',
            parent_id = 2,
            meta      = {'test': 'again'},
            embedding = b'\x00\x00`@',   # [3.5]
        )
        assert doc_d_id == 4

        doc_e_id = q.add_doc(
            text      = 'fifth doc',
            parent_id = 4,
            meta      = {'test': 5},
            embedding = None,
        )
        assert doc_e_id == 5

        with pytest.raises(ValueError):
            q.add_doc(
                text      = 'invalid doc',
                parent_id = 88,  # <-- invalid!
                meta      = None,
                embedding = None,
            )

        assert q._debug_embeddings() == [
            (1, b'\x00\x00\x80?'),
            (2, b'\x00\x00\x00@\x00\x00`@'),
            (3, b'\x00\x00\x00@'),
            (4, b'\x00\x00`@'),
        ]

        assert q._debug_docs() == [
            (1, None, 0, 'first doc', 1, None),
            (2, 1, 1, 'second doc', 2, None),
            (3, None, 0, 'third doc', 3, '{"test": "stuff"}'),
            (4, 2, 2, 'forth doc', 4, '{"test": "again"}'),
            (5, 4, 3, 'fifth doc', None, '{"test": 5}'),
        ]

    db.close()

    # **Re-open** the database
    db = _DB(_DB_PATH)
    with db as q:
        assert q.fetch_doc(1) == {
            'id': 1,
            'parent_id': None,
            'level': 0,
            'text': 'first doc',
            'embedding': [1.0],
            'meta': None,
        }
        assert q.fetch_doc(2) == {
            'id': 2,
            'parent_id': 1,
            'level': 1,
            'text': 'second doc',
            'embedding': [2.0, 3.5],
            'meta': None,
        }
        assert q.fetch_doc(3) == {
            'id': 3,
            'parent_id': None,
            'level': 0,
            'text': 'third doc',
            'embedding': [2.0],
            'meta': {'test': 'stuff'},
        }
        assert q.fetch_doc(4) == {
            'id': 4,
            'parent_id': 2,
            'level': 2,
            'text': 'forth doc',
            'embedding': [3.5],
            'meta': {'test': 'again'},
        }
        assert q.fetch_doc(5) == {
            'id': 5,
            'parent_id': 4,
            'level': 3,
            'text': 'fifth doc',
            'embedding': None,
            'meta': {'test': 5},
        }

        with pytest.raises(KeyError):
            q.fetch_doc(88)

    db.close()

    # **Re-open** the database
    db = _DB(_DB_PATH)
    with db as q:
        q.del_doc(2)
        q.del_doc(4)

        with pytest.raises(KeyError):
            q.del_doc(88)

        assert q._debug_embeddings() == [
            (1, b'\x00\x00\x80?'),
            (3, b'\x00\x00\x00@'),
        ]

        assert q._debug_docs() == [
            (1, None, 0, 'first doc', 1, None),
            (3, None, 0, 'third doc', 3, '{"test": "stuff"}'),
            (5, 4, 3, 'fifth doc', None, '{"test": 5}'),
        ]

    db.close()


def test_rollback():
    # Open the database
    db = _DB(_DB_PATH)
    with db as q:
        assert q._debug_keyval() == {}
        q.set_key('this', 'will persist')
        assert q._debug_keyval() == {
            'this': 'will persist',
        }
    saw_staged_update = False
    with pytest.raises(KeyError):
        with db as q:
            assert q._debug_keyval() == {
                'this': 'will persist',
            }
            q.set_key('this', 'will be rolled back')
            assert q._debug_keyval() == {
                'this': 'will be rolled back',
            }
            saw_staged_update = True
            q.del_key('dne')   # <-- this raises KeyError; since it's uncaught it will `rollback` the transaction!
    assert saw_staged_update
    with db as q:
        # Check that the transaction was rolled back (i.e. `test` wasn't updated).
        assert q._debug_keyval() == {
            'this': 'will persist',
        }
    db.close()


def test_vacuum():
    # Open the database
    db = _DB(_DB_PATH)
    with db as q:
        assert q._debug_keyval() == {}
        q.set_key('this', 'hi!')
        assert q._debug_keyval() == {
            'this': 'hi!',
        }
    db.vacuum()
    db.close()

    # **Re-open** the database
    db = _DB(_DB_PATH)
    with db as q:
        assert q._debug_keyval() == {
            'this': 'hi!',
        }
    db.close()


@pytest.mark.asyncio
async def test_kb_init_and_close():
    # New database; not passing an embedding function.
    kb = KB(_DB_PATH)
    with pytest.raises(RuntimeError):
        # The following will raise because we are in a new database without
        # passing an embedding function.
        await kb.close()

    # New database; this time passing an embedding function.
    kb = KB(_DB_PATH, make_mock_embeddings_func())
    await kb.close()
    assert kb.embedding_func.__name__ == 'mock_embeddings'  # type: ignore

    # Check that the embedding function was stored in the database above!
    db = _DB(_DB_PATH)
    with db as q:
        assert json.loads(q.get_key('embedding_func_params')) == {
            'provider': 'mock',
        }
    db.close()

    # Prev database; it should rebuild the mock embedding func.
    kb = KB(_DB_PATH)
    await kb.close()
    assert kb.embedding_func.__name__ == 'mock_embeddings'  # type: ignore

    # Prev database; override the embedding func.
    kb = KB(_DB_PATH, make_openai_embeddings_func('fake_model', 'fake_apikey'))
    await kb.close()
    assert kb.embedding_func.__name__ == 'openai_embeddings'  # type: ignore

    # Prev database; check that vacuum works.
    kb = KB(_DB_PATH)
    await kb.close(vacuum=True)
    assert kb.embedding_func.__name__ == 'mock_embeddings'  # type: ignore

    # Check that the embedding function is *unchanged*.
    db = _DB(_DB_PATH)
    with db as q:
        assert json.loads(q.get_key('embedding_func_params')) == {
            'provider': 'mock',
        }
    db.close()


@pytest.mark.asyncio
async def test_kb_add_del_doc():
    # New database!
    kb = KB(_DB_PATH, make_mock_embeddings_func())
    assert (await kb.add_doc("first doc")) == 1
    await kb.close()

    # Prev database; let it remember the embedding function!
    kb = KB(_DB_PATH)
    assert (await kb.add_doc("second doc", 1, meta={'a': 'b'})) == 2
    assert (await kb.add_doc("third doc", 1, no_embedding=True)) == 3
    await kb.close()

    # Check the database:
    db = _DB(_DB_PATH)
    with db as q:
        assert json.loads(q.get_key('embedding_func_params')) == {
            'provider': 'mock',
        }
        assert q._debug_embeddings() == [
            (1, b'\x00\x00\x80?\x00\x00\x00\x00\x00\x00\x00\x00'),
            (2, b'\x00\x00\x80?\x00\x00\x00\x00\x00\x00\x00\x00'),
        ]
        assert q._debug_docs() == [
            (1, None, 0, 'first doc', 1, None),
            (2, 1, 1, 'second doc', 2, '{"a": "b"}'),
            (3, 1, 1, 'third doc', None, None),
        ]
    db.close()

    # Prev database; let's delete a document.
    kb = KB(_DB_PATH)
    await kb.del_doc(2)
    await kb.close()

    # Check the database:
    db = _DB(_DB_PATH)
    with db as q:
        assert json.loads(q.get_key('embedding_func_params')) == {
            'provider': 'mock',
        }
        assert q._debug_embeddings() == [
            (1, b'\x00\x00\x80?\x00\x00\x00\x00\x00\x00\x00\x00'),
        ]
        assert q._debug_docs() == [
            (1, None, 0, 'first doc', 1, None),
            (3, 1, 1, 'third doc', None, None),
        ]
    db.close()
