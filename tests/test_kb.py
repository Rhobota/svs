import pytest

import os
import json

from typing import List

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
        assert q.fetch_doc(1, include_embedding=True) == {
            'id': 1,
            'parent_id': None,
            'level': 0,
            'text': 'first doc',
            'embedding': [1.0],
            'meta': None,
        }
        assert q.fetch_doc(2, include_embedding=True) == {
            'id': 2,
            'parent_id': 1,
            'level': 1,
            'text': 'second doc',
            'embedding': [2.0, 3.5],
            'meta': None,
        }
        assert q.fetch_doc(3, include_embedding=True) == {
            'id': 3,
            'parent_id': None,
            'level': 0,
            'text': 'third doc',
            'embedding': [2.0],
            'meta': {'test': 'stuff'},
        }
        assert q.fetch_doc(4, include_embedding=True) == {
            'id': 4,
            'parent_id': 2,
            'level': 2,
            'text': 'forth doc',
            'embedding': [3.5],
            'meta': {'test': 'again'},
        }
        assert q.fetch_doc(5, include_embedding=True) == {
            'id': 5,
            'parent_id': 4,
            'level': 3,
            'text': 'fifth doc',
            'embedding': None,
            'meta': {'test': 5},
        }
        assert q.fetch_doc(4, include_embedding=False) == {
            'id': 4,
            'parent_id': 2,
            'level': 2,
            'text': 'forth doc',
            'embedding': True,
            'meta': {'test': 'again'},
        }
        assert q.fetch_doc(5, include_embedding=False) == {
            'id': 5,
            'parent_id': 4,
            'level': 3,
            'text': 'fifth doc',
            'embedding': False,
            'meta': {'test': 5},
        }

        with pytest.raises(KeyError):
            q.fetch_doc(88, include_embedding=True)

    db.close()

    # **Re-open** the database
    db = _DB(_DB_PATH)
    with db as q:
        with pytest.raises(RuntimeError):
            # This is a parent, so we can't delete it until we delete its children.
            q.del_doc(2)

        q.del_doc(5)
        q.del_doc(4)
        q.del_doc(2)

        with pytest.raises(KeyError):
            q.del_doc(88)

        assert q._debug_embeddings() == [
            (1, b'\x00\x00\x80?'),
            (3, b'\x00\x00\x00@'),
        ]

        assert q._debug_docs() == [
            (1, None, 0, 'first doc', 1, None),
            (3, None, 0, 'third doc', 3, '{"test": "stuff"}'),
        ]

    db.close()

    # **Re-open** the database
    db = _DB(_DB_PATH)
    with db as q:
        q.set_doc_embedding(1, None)
        assert q._debug_embeddings() == [
            (3, b'\x00\x00\x00@'),
        ]
        assert q._debug_docs() == [
            (1, None, 0, 'first doc', None, None),
            (3, None, 0, 'third doc', 3, '{"test": "stuff"}'),
        ]

        q.set_doc_embedding(3, b'\x07')
        assert q._debug_embeddings() == [
            (1, b'\x07'),
        ]
        assert q._debug_docs() == [
            (1, None, 0, 'first doc', None, None),
            (3, None, 0, 'third doc', 1, '{"test": "stuff"}'),
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
        assert q.get_key('schema_version') == 1
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
    async with kb.bulk_add_docs() as add_doc:
        assert (await add_doc("first doc")) == 1
    await kb.close()

    # Prev database; let it remember the embedding function!
    kb = KB(_DB_PATH)
    async with kb.bulk_add_docs() as add_doc:
        assert (await add_doc("second doc", 1, meta={'a': 'b'})) == 2
        assert (await add_doc("third doc", 1, no_embedding=True)) == 3
    with pytest.raises(AssertionError):
        await add_doc('a')  # <-- use outside context manager is not allowed!
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
    async with kb.bulk_del_docs() as del_doc:
        await del_doc(2)
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

    # Prev database; add more documents:
    kb = KB(_DB_PATH)
    async with kb.bulk_add_docs() as add_doc:
        assert (await add_doc("forth doc", 1, meta={'new': 'stuff'})) == 4
        assert (await add_doc("fifth doc", 3, no_embedding=True)) == 5
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
            (3, 1, 1, 'third doc', None, None),
            (4, 1, 1, 'forth doc', 2, '{"new": "stuff"}'),
            (5, 3, 2, 'fifth doc', None, None),
        ]
    db.close()

    # Prev database; delete more documents:
    kb = KB(_DB_PATH)
    async with kb.bulk_del_docs() as del_doc:
        await del_doc(5)
        await del_doc(4)
    with pytest.raises(AssertionError):
        await del_doc(1)  # <-- use outside context manager is not allowed!
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


@pytest.mark.asyncio
async def test_kb_retrieve():
    async def embedding_func(list_of_texts: List[str]) -> List[List[float]]:
        ret: List[List[float]] = []
        for text in list_of_texts:
            if 'first' in text:
                ret.append([1.0, 0.001, 0.0])
            elif 'second' in text:
                ret.append([0.0, 1.0, 0.0001])
            elif 'third' in text:
                ret.append([0.01, 0.0, 1.0])
            elif 'forth' in text:
                ret.append([0.707, 0.707, 0.0])
            else:
                raise ValueError("unexpected doc")
        return ret

    # New database!
    kb = KB(_DB_PATH, embedding_func)
    async with kb.bulk_add_docs() as add_doc:
        assert (await add_doc("third doc")) == 1
        assert (await add_doc("first doc")) == 2
        assert (await add_doc("second doc")) == 3
    await kb.close()

    # Retrieve!
    kb = KB(_DB_PATH, embedding_func)

    docs = await kb.retrieve('... first ...', n=3)
    assert len(docs) == 3
    assert docs[0]['doc'] and docs[0]['doc']['text'] == 'first doc'
    assert docs[1]['doc'] and docs[1]['doc']['text'] == 'third doc'
    assert docs[2]['doc'] and docs[2]['doc']['text'] == 'second doc'

    docs = await kb.retrieve('... second ...', n=3)
    assert len(docs) == 3
    assert docs[0]['doc'] and docs[0]['doc']['text'] == 'second doc'
    assert docs[1]['doc'] and docs[1]['doc']['text'] == 'first doc'
    assert docs[2]['doc'] and docs[2]['doc']['text'] == 'third doc'

    docs = await kb.retrieve('... third ...', n=3)
    assert len(docs) == 3
    assert docs[0]['doc'] and docs[0]['doc']['text'] == 'third doc'
    assert docs[1]['doc'] and docs[1]['doc']['text'] == 'first doc'
    assert docs[2]['doc'] and docs[2]['doc']['text'] == 'second doc'

    await kb.close()

    # Add and retrieve (i.e. test invalidating the embeddings)
    kb = KB(_DB_PATH, embedding_func)
    await kb.load()

    async with kb.bulk_add_docs() as add_doc:
        assert (await add_doc('forth doc')) == 4

    docs = await kb.retrieve('... forth ...', n=1)
    assert len(docs) == 1
    assert docs[0]['doc'] and docs[0]['doc']['text'] == 'forth doc'

    await kb.close()

    # Delete and retrieve (i.e. test invalidating the embeddings)
    kb = KB(_DB_PATH, embedding_func)
    await kb.load()

    async with kb.bulk_del_docs() as del_doc:
        await del_doc(1)
        await del_doc(2)
        await del_doc(4)

    docs = await kb.retrieve('... forth ...', n=1)
    assert len(docs) == 1
    assert docs[0]['doc'] and docs[0]['doc']['text'] == 'second doc'

    await kb.close()


@pytest.mark.asyncio
async def test_kb_vector_magnitude():
    async def embedding_func_1(list_of_texts: List[str]) -> List[List[float]]:
        return [
            [1.0, 0.1, 0.0]  # <-- magnitude too large
            for _ in list_of_texts
        ]
    async def embedding_func_2(list_of_texts: List[str]) -> List[List[float]]:
        return [
            [0.99, 0.0, 0.0]  # <-- magnitude too small
            for _ in list_of_texts
        ]

    # Test magnitude too large:
    kb = KB(_DB_PATH, embedding_func_1)
    with pytest.raises(ValueError):
        async with kb.bulk_add_docs() as add_doc:
            await add_doc("...")
    await kb.close()

    # Test magnitude too small:
    kb = KB(_DB_PATH, embedding_func_2)
    with pytest.raises(ValueError):
        async with kb.bulk_add_docs() as add_doc:
            await add_doc("...")
    await kb.close()
