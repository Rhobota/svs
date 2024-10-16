import pytest

import os
import json
import gzip

import numpy as np

import networkx as nx  # type: ignore

from typing import List

from svs.embeddings import (
    make_mock_embeddings_func,
    make_openai_embeddings_func,
)

from svs.kb import (
    AsyncKB,
    KB,
    _DB,
    SQLITE_IS_STRICT,
)

from svs.util import delete_file_if_exists


_DB_PATH = './testdb.sqlite'


@pytest.fixture(autouse=True)
def clear_database():
    def clear():
        delete_file_if_exists(_DB_PATH)
        delete_file_if_exists(f'{_DB_PATH}.gz')

    clear()
    yield
    clear()


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
        assert q.count_docs() == 0

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

        assert q.count_docs() == 5

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

        assert q.fetch_doc_children(2, include_embedding=False) == [
            {
                'id': 4,
                'parent_id': 2,
                'level': 2,
                'text': 'forth doc',
                'embedding': True,
                'meta': {'test': 'again'},
            },
        ]

        new_doc_id = q.add_doc(
            text      = 'sixth doc',
            parent_id = 2,
            meta      = {'test': 6},
            embedding = b'\x07',
        )
        assert new_doc_id == 6

        assert q.fetch_doc_children(2, include_embedding=False) == [
            {
                'id': 4,
                'parent_id': 2,
                'level': 2,
                'text': 'forth doc',
                'embedding': True,
                'meta': {'test': 'again'},
            },
            {
                'id': 6,
                'parent_id': 2,
                'level': 2,
                'text': 'sixth doc',
                'embedding': True,
                'meta': {'test': 6},
            },
        ]

        assert q.fetch_docs_at_level(0, include_embedding=False) == [
            {
                'id': 1,
                'parent_id': None,
                'level': 0,
                'text': 'first doc',
                'embedding': True,
                'meta': None,
            },
            {
                'id': 3,
                'parent_id': None,
                'level': 0,
                'text': 'third doc',
                'embedding': True,
                'meta': {'test': 'stuff'},
            },
        ]

        assert q.fetch_docs_at_level(1, include_embedding=True) == [
            {
                'id': 2,
                'parent_id': 1,
                'level': 1,
                'text': 'second doc',
                'embedding': [2.0, 3.5],
                'meta': None,
            },
        ]

        assert q.fetch_docs_at_level(2, include_embedding=False) == [
            {
                'id': 4,
                'parent_id': 2,
                'level': 2,
                'text': 'forth doc',
                'embedding': True,
                'meta': {'test': 'again'},
            },
            {
                'id': 6,
                'parent_id': 2,
                'level': 2,
                'text': 'sixth doc',
                'embedding': True,
                'meta': {'test': 6},
            },
        ]

        assert q.fetch_doc_with_emb_id(1) == 1
        assert q.fetch_doc_with_emb_id(2) == 2
        assert q.fetch_doc_with_emb_id(3) == 3
        assert q.fetch_doc_with_emb_id(4) == 4
        assert q.fetch_doc_with_emb_id(5) == 6
        with pytest.raises(KeyError):
            q.fetch_doc_with_emb_id(6)

    db.close()

    # **Re-open** the database
    db = _DB(_DB_PATH)
    with db as q:
        with pytest.raises(RuntimeError):
            # This is a parent, so we can't delete it until we delete its children.
            q.del_doc(2)

        q.del_doc(6)
        q.del_doc(5)
        q.del_doc(4)
        q.del_doc(2)

        assert q.count_docs() == 2

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


def test_edge_table():
    # Open the database
    db = _DB(_DB_PATH)
    with db as q:
        assert q.count_docs() == 0
        assert q.count_edges() == 0

        doc_a_id = q.add_doc(
            text      = 'first doc',
            parent_id = None,
            meta      = None,
            embedding = None,
        )
        assert doc_a_id == 1

        doc_b_id = q.add_doc(
            text      = 'second doc',
            parent_id = None,
            meta      = None,
            embedding = None,
        )
        assert doc_b_id == 2

        doc_c_id = q.add_doc(
            text      = 'third doc',
            parent_id = None,
            meta      = {'test': 'stuff'},
            embedding = None,
        )
        assert doc_c_id == 3

        doc_d_id = q.add_doc(
            text      = 'forth doc',
            parent_id = None,
            meta      = {'test': 'again'},
            embedding = None,
        )
        assert doc_d_id == 4

        doc_e_id = q.add_doc(
            text      = 'fifth doc',
            parent_id = None,
            meta      = {'test': 5},
            embedding = None,
        )
        assert doc_e_id == 5

        edge_type_1 = q.add_doc(
            text      = 'edge type 1',
            parent_id = None,
            meta      = None,
            embedding = None,
        )
        assert edge_type_1 == 6

        edge_type_2 = q.add_doc(
            text      = 'edge type 2',
            parent_id = None,
            meta      = None,
            embedding = None,
        )
        assert edge_type_2 == 7

        assert q.count_docs() == 7
        assert q.count_edges() == 0

        assert q._debug_docs() == [
            (1, None, 0, 'first doc', None, None),
            (2, None, 0, 'second doc', None, None),
            (3, None, 0, 'third doc', None, '{"test": "stuff"}'),
            (4, None, 0, 'forth doc', None, '{"test": "again"}'),
            (5, None, 0, 'fifth doc', None, '{"test": 5}'),
            (6, None, 0, 'edge type 1', None, None),
            (7, None, 0, 'edge type 2', None, None),
        ]

        assert q._debug_edges() == []

        edge_1_id = q.add_edge(
            doc_b_id,
            doc_d_id,
            edge_type_1,
            weight = None,
        )
        assert edge_1_id == 1

        edge_2_id = q.add_edge(
            doc_b_id,
            doc_d_id,
            edge_type_2,
            weight = None,
        )
        assert edge_2_id == 2

        edge_3_id = q.add_edge(
            doc_a_id,
            doc_d_id,
            edge_type_1,
            weight = 0.5,
        )
        assert edge_3_id == 3

        edge_4_id = q.add_edge(
            doc_a_id,
            doc_c_id,
            edge_type_2,
            weight = 1.5,
        )
        assert edge_4_id == 4

        edge_5_id = q.add_directed_edge(
            doc_b_id,
            doc_c_id,
            edge_type_1,
            weight = None,
        )
        assert edge_5_id == 5

        edge_6_id = q.add_directed_edge(
            doc_b_id,
            doc_e_id,
            edge_type_2,
            weight = 2.5,
        )
        assert edge_6_id == 6

        assert q.count_docs() == 7
        assert q.count_edges() == 6

        assert q._debug_docs() == [
            (1, None, 0, 'first doc', None, None),
            (2, None, 0, 'second doc', None, None),
            (3, None, 0, 'third doc', None, '{"test": "stuff"}'),
            (4, None, 0, 'forth doc', None, '{"test": "again"}'),
            (5, None, 0, 'fifth doc', None, '{"test": 5}'),
            (6, None, 0, 'edge type 1', None, None),
            (7, None, 0, 'edge type 2', None, None),
        ]

        assert q._debug_edges() == [
            (1, 2, 4, 6, None, 0),
            (2, 2, 4, 7, None, 0),
            (3, 1, 4, 6, 0.5, 0),
            (4, 1, 3, 7, 1.5, 0),
            (5, 2, 3, 6, None, 1),
            (6, 2, 5, 7, 2.5, 1),
        ]

        with pytest.raises(RuntimeError):
            # Below attempts to insert a duplicate edge!
            q.add_edge(
                doc_b_id,
                doc_e_id,
                edge_type_2,
                weight=None,
            )

    db.close()

    # **Re-open** the database
    db = _DB(_DB_PATH)
    with db as q:
        assert q.count_docs() == 7
        assert q.count_edges() == 6

        assert q._debug_docs() == [
            (1, None, 0, 'first doc', None, None),
            (2, None, 0, 'second doc', None, None),
            (3, None, 0, 'third doc', None, '{"test": "stuff"}'),
            (4, None, 0, 'forth doc', None, '{"test": "again"}'),
            (5, None, 0, 'fifth doc', None, '{"test": 5}'),
            (6, None, 0, 'edge type 1', None, None),
            (7, None, 0, 'edge type 2', None, None),
        ]

        assert q._debug_edges() == [
            (1, 2, 4, 6, None, 0),
            (2, 2, 4, 7, None, 0),
            (3, 1, 4, 6, 0.5, 0),
            (4, 1, 3, 7, 1.5, 0),
            (5, 2, 3, 6, None, 1),
            (6, 2, 5, 7, 2.5, 1),
        ]

        with pytest.raises(RuntimeError):
            # Below attempts to insert a duplicate edge!
            q.add_directed_edge(
                doc_a_id,
                doc_c_id,
                edge_type_2,
                weight=None,
            )

        graph = q.build_networkx_graph(multigraph = True)
        assert isinstance(graph, nx.MultiDiGraph)
        assert set(graph.nodes()) == {1, 2, 3, 4, 5}
        edge_set = set()
        for a, b, attrs in graph.edges(data=True):
            edge_set.add((a, b, attrs.get('edge_doc'), attrs.get('weight')))
        assert edge_set == {
            (2, 4, 6, None),
            (2, 4, 7, None),
            (1, 4, 6, 0.5),
            (1, 3, 7, 1.5),
            (2, 3, 6, None),
            (2, 5, 7, 2.5),
            (4, 2, 6, None),
            (4, 2, 7, None),
            (4, 1, 6, 0.5),
            (3, 1, 7, 1.5),
        }

    db.close()

    # **Re-open** the database
    db = _DB(_DB_PATH)
    with db as q:
        assert q.count_docs() == 7
        assert q.count_edges() == 6

        q.del_edge(edge_2_id)

        assert q.count_docs() == 7
        assert q.count_edges() == 5

        assert q._debug_edges() == [
            (1, 2, 4, 6, None, 0),
            (3, 1, 4, 6, 0.5, 0),
            (4, 1, 3, 7, 1.5, 0),
            (5, 2, 3, 6, None, 1),
            (6, 2, 5, 7, 2.5, 1),
        ]

        graph = q.build_networkx_graph(multigraph = False)
        assert isinstance(graph, nx.DiGraph)
        assert set(graph.nodes()) == {1, 2, 3, 4, 5}
        edge_set = set()
        for a, b, attrs in graph.edges(data=True):
            edge_set.add((a, b, attrs.get('edge_doc'), attrs.get('weight')))
        assert edge_set == {
            (1, 3, 7, 1.5),
            (1, 4, 6, 0.5),
            (2, 3, 6, None),
            (2, 4, 6, None),
            (2, 5, 7, 2.5),
            (3, 1, 7, 1.5),
            (4, 1, 6, 0.5),
            (4, 2, 6, None),
        }

        q.del_doc(doc_a_id)
        q.del_doc(doc_e_id)

        assert q.count_docs() == 5
        assert q.count_edges() == 2

        assert q._debug_docs() == [
            (2, None, 0, 'second doc', None, None),
            (3, None, 0, 'third doc', None, '{"test": "stuff"}'),
            (4, None, 0, 'forth doc', None, '{"test": "again"}'),
            (6, None, 0, 'edge type 1', None, None),
            (7, None, 0, 'edge type 2', None, None),
        ]

        assert q._debug_edges() == [
            (1, 2, 4, 6, None, 0),
            (5, 2, 3, 6, None, 1),
        ]

        graph = q.build_networkx_graph(multigraph = False)
        assert isinstance(graph, nx.DiGraph)
        assert set(graph.nodes()) == {2, 3, 4}
        edge_set = set()
        for a, b, attrs in graph.edges(data=True):
            edge_set.add((a, b, attrs.get('edge_doc'), attrs.get('weight')))
        assert edge_set == {
            (2, 3, 6, None),
            (2, 4, 6, None),
            (4, 2, 6, None),
        }

        q.del_edge(edge_5_id)

        assert q.count_docs() == 5
        assert q.count_edges() == 1

        graph = q.build_networkx_graph(multigraph = False)
        assert isinstance(graph, nx.Graph)
        assert set(graph.nodes()) == {2, 4}
        edge_set = set()
        for a, b, attrs in graph.edges(data=True):
            a, b = min(a, b), max(a, b)
            edge_set.add((a, b, attrs.get('edge_doc'), attrs.get('weight')))
        assert edge_set == {
            (2, 4, 6, None),
        }

    db.close()

    # **Re-open** the database
    db = _DB(_DB_PATH)
    with db as q:
        assert q.count_docs() == 5
        assert q.count_edges() == 1

        assert q._debug_docs() == [
            (2, None, 0, 'second doc', None, None),
            (3, None, 0, 'third doc', None, '{"test": "stuff"}'),
            (4, None, 0, 'forth doc', None, '{"test": "again"}'),
            (6, None, 0, 'edge type 1', None, None),
            (7, None, 0, 'edge type 2', None, None),
        ]

        assert q._debug_edges() == [
            (1, 2, 4, 6, None, 0),
        ]

    db.close()


def test_embedding_matrix():
    # Open the database
    db = _DB(_DB_PATH)
    with db as q:
        doc_a_id = q.add_doc(
            text      = 'first doc',
            parent_id = None,
            meta      = None,
            embedding = b'\x00\x00\x80?\x00\x00`@',  # [1.0, 3.5]
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
            embedding = b'\x00\x00\x00@\x00\x00\x80?', # [2.0, 1.0]
        )
        assert doc_c_id == 3

        doc_d_id = q.add_doc(
            text      = 'forth doc',
            parent_id = 2,
            meta      = {'test': 'again'},
            embedding = b'\x00\x00`@\x00\x00\x80@',   # [3.5, 4.0]
        )
        assert doc_d_id == 4

        embeddings_matrix, emb_id_lookup = q.build_embeddings_matrix()
        assert (embeddings_matrix == np.array([
            [1.0, 3.5],
            [2.0, 3.5],
            [2.0, 1.0],
            [3.5, 4.0],
        ])).all()
        assert (emb_id_lookup == np.array([1, 2, 3, 4])).all()

        q.del_doc(3)

        embeddings_matrix, emb_id_lookup = q.build_embeddings_matrix()
        assert (embeddings_matrix == np.array([
            [1.0, 3.5],
            [2.0, 3.5],
            [3.5, 4.0],
        ])).all()
        assert (emb_id_lookup == np.array([1, 2, 4])).all()

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


@pytest.mark.asyncio
async def test_rollback_async():
    # Open the database
    db = _DB(_DB_PATH)
    async with db as q:
        assert q._debug_keyval() == {}
        q.set_key('this', 'will persist')
        assert q._debug_keyval() == {
            'this': 'will persist',
        }
    saw_staged_update = False
    with pytest.raises(KeyError):
        async with db as q:
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
    async with db as q:
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


def test_schema_version_new_database():
    # Open the database
    db = _DB(_DB_PATH)
    db.check_or_set_schema_version()  # <-- will `set_key` since this is a new database
    with db as q:
        assert q.get_key('schema_version') == 1
    db.close()


def test_schema_version_same_version():
    # Open the database
    db = _DB(_DB_PATH)
    with db as q:
        q.set_key('schema_version', 1)  # <-- current version
    db.check_or_set_schema_version()  # <-- will *not* raise; version is current
    db.close()


def test_schema_version_bad_version():
    # Open the database
    db = _DB(_DB_PATH)
    with db as q:
        q.set_key('schema_version', 99)  # <-- BAD VERSION
    with pytest.raises(RuntimeError):
        db.check_or_set_schema_version()
    db.close()


@pytest.mark.asyncio
async def test_asynckb_init_and_close():
    # New database; not passing an embedding function.
    kb = AsyncKB(_DB_PATH)
    with pytest.raises(RuntimeError):
        # The following will raise because we are in a new database without
        # passing an embedding function.
        await kb.close()

    # New database; this time passing an embedding function.
    kb = AsyncKB(_DB_PATH, make_mock_embeddings_func())
    await kb.load()
    assert kb.embedding_func.__name__ == 'mock_embeddings'  # type: ignore
    await kb.close()

    # Check that the embedding function was stored in the database above!
    db = _DB(_DB_PATH)
    with db as q:
        assert json.loads(q.get_key('embedding_func_params')) == {
            'provider': 'mock',
        }
        assert q.get_key('schema_version') == 1
    db.close()

    # Prev database; it should rebuild the mock embedding func.
    kb = AsyncKB(_DB_PATH)
    await kb.load()
    assert kb.embedding_func.__name__ == 'mock_embeddings'  # type: ignore
    await kb.close()
    assert kb.embedding_func is None   # <-- cleared on close

    # Prev database; override the embedding func.
    kb = AsyncKB(_DB_PATH, make_openai_embeddings_func('fake_model', 'fake_apikey'))
    await kb.load()
    assert kb.embedding_func.__name__ == 'openai_embeddings'  # type: ignore
    await kb.close()

    # Prev database; check that vacuum works.
    kb = AsyncKB(_DB_PATH)
    await kb.load()
    assert kb.embedding_func.__name__ == 'mock_embeddings'  # type: ignore
    await kb.close(vacuum=True)

    # Check that the embedding function is *unchanged*.
    db = _DB(_DB_PATH)
    with db as q:
        assert json.loads(q.get_key('embedding_func_params')) == {
            'provider': 'mock',
        }
        assert q.get_key('schema_version') == 1
    db.close()

    # Prev database; check that `also_gzip` works.
    gz_path = f'{_DB_PATH}.gz'
    delete_file_if_exists(gz_path)
    assert not os.path.exists(gz_path)
    kb = AsyncKB(_DB_PATH)
    await kb.load()
    assert kb.embedding_func.__name__ == 'mock_embeddings'  # type: ignore
    await kb.close(also_gzip=True)
    assert os.path.exists(gz_path)
    with gzip.open(gz_path, 'rb') as f:
        content1 = f.read()
    with open(_DB_PATH, 'rb') as f:
        content2 = f.read()
    assert content1 == content2


@pytest.mark.asyncio
async def test_asynckb_add_del_doc():
    # New database!
    kb = AsyncKB(_DB_PATH, make_mock_embeddings_func())
    async with kb.bulk_add_docs() as add_doc:
        assert (await add_doc("first doc")) == 1
    await kb.close()

    # Prev database; let it remember the embedding function!
    kb = AsyncKB(_DB_PATH)
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
    kb = AsyncKB(_DB_PATH)
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
    kb = AsyncKB(_DB_PATH)
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

    # Prev database; bulk query:
    kb = AsyncKB(_DB_PATH)
    async with kb.bulk_query_docs() as q:
        assert (await q.count()) == 4
        assert (await q.query_doc(1)) == {
            'id': 1,
            'text': 'first doc',
            'parent_id': None,
            'level': 0,
            'meta': None,
            'embedding': True,
        }
        assert (await q.query_children(1)) == [
            {
                'id': 3,
                'text': 'third doc',
                'parent_id': 1,
                'level': 1,
                'meta': None,
                'embedding': False,
            },
            {
                'id': 4,
                'text': 'forth doc',
                'parent_id': 1,
                'level': 1,
                'meta': {'new': 'stuff'},
                'embedding': True,
            },
        ]
        assert (await q.query_level(1)) == [
            {
                'id': 3,
                'text': 'third doc',
                'parent_id': 1,
                'level': 1,
                'meta': None,
                'embedding': False,
            },
            {
                'id': 4,
                'text': 'forth doc',
                'parent_id': 1,
                'level': 1,
                'meta': {'new': 'stuff'},
                'embedding': True,
            },
        ]
        nodes = []
        async for node in q.dfs_traversal():
            nodes.append(node)
        assert nodes == [
            {
                'id': 1,
                'text': 'first doc',
                'parent_id': None,
                'level': 0,
                'meta': None,
                'embedding': True,
            },
            {
                'id': 3,
                'text': 'third doc',
                'parent_id': 1,
                'level': 1,
                'meta': None,
                'embedding': False,
            },
            {
                'id': 5,
                'text': 'fifth doc',
                'parent_id': 3,
                'level': 2,
                'meta': None,
                'embedding': False,
            },
            {
                'id': 4,
                'text': 'forth doc',
                'parent_id': 1,
                'level': 1,
                'meta': {'new': 'stuff'},
                'embedding': True,
            },
        ]
    await kb.close()

    # Prev database; delete more documents:
    kb = AsyncKB(_DB_PATH)
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

    # Prev database; check that `force_fresh_db` works.
    kb = AsyncKB(_DB_PATH, make_mock_embeddings_func(), force_fresh_db=True)
    await kb.load()
    await kb.close()

    # Check the database (it should be empty now from the `force_fresh_db` above):
    db = _DB(_DB_PATH)
    with db as q:
        assert json.loads(q.get_key('embedding_func_params')) == {
            'provider': 'mock',
        }
        assert q._debug_embeddings() == []
        assert q._debug_docs() == []
    db.close()


@pytest.mark.asyncio
async def test_asynckb_retrieve_et_al():
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
    kb = AsyncKB(_DB_PATH, embedding_func)
    async with kb.bulk_add_docs() as add_doc:
        assert (await add_doc("third doc")) == 1
        assert (await add_doc("first doc")) == 2
        assert (await add_doc("second doc")) == 3
    await kb.close()

    # Retrieve!
    kb = AsyncKB(_DB_PATH, embedding_func)

    docs = await kb.retrieve('... first ...', n=3)
    assert len(docs) == 3
    assert docs[0]['doc']['text'] == 'first doc'
    assert docs[1]['doc']['text'] == 'third doc'
    assert docs[2]['doc']['text'] == 'second doc'

    docs = await kb.retrieve('... second ...', n=3)
    assert len(docs) == 3
    assert docs[0]['doc']['text'] == 'second doc'
    assert docs[1]['doc']['text'] == 'first doc'
    assert docs[2]['doc']['text'] == 'third doc'

    docs = await kb.retrieve('... third ...', n=3)
    assert len(docs) == 3
    assert docs[0]['doc']['text'] == 'third doc'
    assert docs[1]['doc']['text'] == 'first doc'
    assert docs[2]['doc']['text'] == 'second doc'

    await kb.close()

    # Pairwise scores:
    kb = AsyncKB(_DB_PATH, embedding_func)

    records = await kb.document_top_pairwise_scores(n = 2)
    assert len(records) == 2

    _, doc_1, doc_2 = records[0]
    assert doc_1['id'] == 1
    assert doc_2['id'] == 2

    _, doc_1, doc_2 = records[1]
    assert doc_1['id'] == 2
    assert doc_2['id'] == 3

    await kb.close()

    # Add and retrieve (i.e. test invalidating the embeddings)
    kb = AsyncKB(_DB_PATH, embedding_func)
    await kb.load()

    docs = await kb.retrieve('... forth ...', n=1)
    assert len(docs) == 1
    assert docs[0]['doc']['text'] == 'first doc'

    async with kb.bulk_add_docs() as add_doc:
        assert (await add_doc('forth doc')) == 4

    docs = await kb.retrieve('... forth ...', n=1)
    assert len(docs) == 1
    assert docs[0]['doc']['text'] == 'forth doc'

    await kb.close()

    # Graph interface:
    kb = AsyncKB(_DB_PATH, embedding_func)
    await kb.load()

    async with kb.bulk_graph_update() as q:
        assert (await q.count_edges()) == 0
        await q.add_directed_edge(1, 3, 4)
        await q.add_edge(1, 2, 4)
        assert (await q.count_edges()) == 2
        graph = await q.build_networkx_graph(multigraph=False)
        assert isinstance(graph, nx.DiGraph)
        await q.del_edge(1)
        assert (await q.count_edges()) == 1

    await kb.close()

    # Delete and retrieve (i.e. test invalidating the embeddings)
    kb = AsyncKB(_DB_PATH, embedding_func)
    await kb.load()

    docs = await kb.retrieve('... forth ...', n=1)
    assert len(docs) == 1
    assert docs[0]['doc']['text'] == 'forth doc'

    async with kb.bulk_del_docs() as del_doc:
        await del_doc(1)
        await del_doc(2)
        await del_doc(4)

    docs = await kb.retrieve('... forth ...', n=1)
    assert len(docs) == 1
    assert docs[0]['doc']['text'] == 'second doc'

    await kb.close()


@pytest.mark.asyncio
async def test_asynckb_vector_magnitude():
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
    kb = AsyncKB(_DB_PATH, embedding_func_1)
    with pytest.raises(ValueError):
        async with kb.bulk_add_docs() as add_doc:
            await add_doc("...")
    await kb.close()

    # Test magnitude too small:
    kb = AsyncKB(_DB_PATH, embedding_func_2)
    with pytest.raises(ValueError):
        async with kb.bulk_add_docs() as add_doc:
            await add_doc("...")
    await kb.close()


@pytest.mark.asyncio
async def test_asynckb_keyval_interface():
    # New database!
    kb = AsyncKB(_DB_PATH, make_mock_embeddings_func())
    async with kb.bulk_keyval_update() as q:
        assert (await q.count()) == 0
        assert not (await q.has('a'))
        assert not (await q.has('b'))
        with pytest.raises(KeyError):
            await q.get('a')
        with pytest.raises(KeyError):
            await q.get('b')
        with pytest.raises(ValueError):
            await q.get('a', ValueError)
        with pytest.raises(ValueError):
            await q.get('b', ValueError)
        assert (await q.get('a', 'a_default')) == 'a_default'
        assert (await q.get('b', 'b_default')) == 'b_default'

        await q.set('b', 77)

        assert (await q.count()) == 1
        assert not (await q.has('a'))
        assert (await q.has('b'))
        with pytest.raises(KeyError):
            await q.get('a')
        assert (await q.get('b')) == 77
        with pytest.raises(ValueError):
            await q.get('a', ValueError)
        assert (await q.get('b', ValueError)) == 77
        assert (await q.get('a', 'a_default')) == 'a_default'
        assert (await q.get('b', 'b_default')) == 77

        await q.set('b', 'new_val')

        assert (await q.count()) == 1
        assert not (await q.has('a'))
        assert (await q.has('b'))
        with pytest.raises(KeyError):
            await q.get('a')
        assert (await q.get('b')) == 'new_val'
        with pytest.raises(ValueError):
            await q.get('a', ValueError)
        assert (await q.get('b', ValueError)) == 'new_val'
        assert (await q.get('a', 'a_default')) == 'a_default'
        assert (await q.get('b', 'b_default')) == 'new_val'

    await kb.close()

    # Prev database; let it remember the embedding function!
    kb = AsyncKB(_DB_PATH)
    async with kb.bulk_keyval_update() as q:
        assert (await q.count()) == 1
        assert not (await q.has('a'))
        assert (await q.has('b'))
        with pytest.raises(KeyError):
            await q.get('a')
        assert (await q.get('b')) == 'new_val'
        with pytest.raises(ValueError):
            await q.get('a', ValueError)
        assert (await q.get('b', ValueError)) == 'new_val'
        assert (await q.get('a', 'a_default')) == 'a_default'
        assert (await q.get('b', 'b_default')) == 'new_val'

        with pytest.raises(KeyError):
            await q.remove('dne')

        await q.remove('b')

        assert (await q.count()) == 0
        assert not (await q.has('a'))
        assert not (await q.has('b'))
        with pytest.raises(KeyError):
            await q.get('a')
        with pytest.raises(KeyError):
            await q.get('b')
        with pytest.raises(ValueError):
            await q.get('a', ValueError)
        with pytest.raises(ValueError):
            await q.get('b', ValueError)
        assert (await q.get('a', 'a_default')) == 'a_default'
        assert (await q.get('b', 'b_default')) == 'b_default'

    await kb.close()

    # Prev database; let it remember the embedding function!
    kb = AsyncKB(_DB_PATH)
    async with kb.bulk_keyval_update() as q:
        assert (await q.count()) == 0
        assert not (await q.has('a'))
        assert not (await q.has('b'))
        with pytest.raises(KeyError):
            await q.get('a')
        with pytest.raises(KeyError):
            await q.get('b')
        with pytest.raises(ValueError):
            await q.get('a', ValueError)
        with pytest.raises(ValueError):
            await q.get('b', ValueError)
        assert (await q.get('a', 'a_default')) == 'a_default'
        assert (await q.get('b', 'b_default')) == 'b_default'

        await q.set('reason', 'because')
        await q.set('answer', 42)
        await q.set('age', 87.5)

        assert set([v async for v in q.items()]) == {
            ('reason', 'because'),
            ('answer', 42),
            ('age', 87.5),
        }

    await kb.close()


def test_kb_init_and_close():
    # New database; not passing an embedding function.
    with pytest.raises(RuntimeError):
        # The following will raise because we are in a new database without
        # passing an embedding function.
        kb = KB(_DB_PATH)

    # New database; this time passing an embedding function.
    kb = KB(_DB_PATH, make_mock_embeddings_func())
    assert kb.embedding_func.__name__ == 'mock_embeddings'  # type: ignore
    kb.close()

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
    assert kb.embedding_func.__name__ == 'mock_embeddings'  # type: ignore
    kb.close()
    assert kb.embedding_func is None   # <-- cleared on close

    # Prev database; override the embedding func.
    kb = KB(_DB_PATH, make_openai_embeddings_func('fake_model', 'fake_apikey'))
    assert kb.embedding_func.__name__ == 'openai_embeddings'  # type: ignore
    kb.close()

    # Prev database; check that vacuum works.
    kb = KB(_DB_PATH)
    assert kb.embedding_func.__name__ == 'mock_embeddings'  # type: ignore
    kb.close(vacuum=True)

    # Check that the embedding function is *unchanged*.
    db = _DB(_DB_PATH)
    with db as q:
        assert json.loads(q.get_key('embedding_func_params')) == {
            'provider': 'mock',
        }
        assert q.get_key('schema_version') == 1
    db.close()

    # Prev database; check that `also_gzip` works.
    gz_path = f'{_DB_PATH}.gz'
    delete_file_if_exists(gz_path)
    assert not os.path.exists(gz_path)
    kb = KB(_DB_PATH)
    assert kb.embedding_func.__name__ == 'mock_embeddings'  # type: ignore
    kb.close(also_gzip=True)
    assert os.path.exists(gz_path)
    with gzip.open(gz_path, 'rb') as f:
        content1 = f.read()
    with open(_DB_PATH, 'rb') as f:
        content2 = f.read()
    assert content1 == content2


def test_kb_add_del_doc():
    # New database!
    kb = KB(_DB_PATH, make_mock_embeddings_func())
    with kb.bulk_add_docs() as add_doc:
        assert add_doc("first doc") == 1
    kb.close()

    # Prev database; let it remember the embedding function!
    kb = KB(_DB_PATH)
    with kb.bulk_add_docs() as add_doc:
        assert add_doc("second doc", 1, meta={'a': 'b'}) == 2
        assert add_doc("third doc", 1, no_embedding=True) == 3
    with pytest.raises(AssertionError):
        add_doc('a')  # <-- use outside context manager is not allowed!
    kb.close()

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
    with kb.bulk_del_docs() as del_doc:
        del_doc(2)
    kb.close()

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
    with kb.bulk_add_docs() as add_doc:
        assert add_doc("forth doc", 1, meta={'new': 'stuff'}) == 4
        assert add_doc("fifth doc", 3, no_embedding=True) == 5
    kb.close()

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

    # Prev database; bulk query:
    kb = KB(_DB_PATH)
    with kb.bulk_query_docs() as q:
        assert q.count() == 4
        assert q.query_doc(1) == {
            'id': 1,
            'text': 'first doc',
            'parent_id': None,
            'level': 0,
            'meta': None,
            'embedding': True,
        }
        assert q.query_children(1) == [
            {
                'id': 3,
                'text': 'third doc',
                'parent_id': 1,
                'level': 1,
                'meta': None,
                'embedding': False,
            },
            {
                'id': 4,
                'text': 'forth doc',
                'parent_id': 1,
                'level': 1,
                'meta': {'new': 'stuff'},
                'embedding': True,
            },
        ]
        assert q.query_level(1) == [
            {
                'id': 3,
                'text': 'third doc',
                'parent_id': 1,
                'level': 1,
                'meta': None,
                'embedding': False,
            },
            {
                'id': 4,
                'text': 'forth doc',
                'parent_id': 1,
                'level': 1,
                'meta': {'new': 'stuff'},
                'embedding': True,
            },
        ]
        nodes = []
        for node in q.dfs_traversal():
            nodes.append(node)
        assert nodes == [
            {
                'id': 1,
                'text': 'first doc',
                'parent_id': None,
                'level': 0,
                'meta': None,
                'embedding': True,
            },
            {
                'id': 3,
                'text': 'third doc',
                'parent_id': 1,
                'level': 1,
                'meta': None,
                'embedding': False,
            },
            {
                'id': 5,
                'text': 'fifth doc',
                'parent_id': 3,
                'level': 2,
                'meta': None,
                'embedding': False,
            },
            {
                'id': 4,
                'text': 'forth doc',
                'parent_id': 1,
                'level': 1,
                'meta': {'new': 'stuff'},
                'embedding': True,
            },
        ]
    kb.close()

    # Prev database; delete more documents:
    kb = KB(_DB_PATH)
    with kb.bulk_del_docs() as del_doc:
        del_doc(5)
        del_doc(4)
    with pytest.raises(AssertionError):
        del_doc(1)  # <-- use outside context manager is not allowed!
    kb.close()

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

    # Prev database; check that `force_fresh_db` works.
    kb = KB(_DB_PATH, make_mock_embeddings_func(), force_fresh_db=True)
    kb.close()

    # Check the database (it should be empty now from the `force_fresh_db` above):
    db = _DB(_DB_PATH)
    with db as q:
        assert json.loads(q.get_key('embedding_func_params')) == {
            'provider': 'mock',
        }
        assert q._debug_embeddings() == []
        assert q._debug_docs() == []
    db.close()


def test_kb_retrieve_et_al():
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
    with kb.bulk_add_docs() as add_doc:
        assert add_doc("third doc") == 1
        assert add_doc("first doc") == 2
        assert add_doc("second doc") == 3
    kb.close()

    # Retrieve!
    kb = KB(_DB_PATH, embedding_func)

    docs = kb.retrieve('... first ...', n=3)
    assert len(docs) == 3
    assert docs[0]['doc']['text'] == 'first doc'
    assert docs[1]['doc']['text'] == 'third doc'
    assert docs[2]['doc']['text'] == 'second doc'

    docs = kb.retrieve('... second ...', n=3)
    assert len(docs) == 3
    assert docs[0]['doc']['text'] == 'second doc'
    assert docs[1]['doc']['text'] == 'first doc'
    assert docs[2]['doc']['text'] == 'third doc'

    docs = kb.retrieve('... third ...', n=3)
    assert len(docs) == 3
    assert docs[0]['doc']['text'] == 'third doc'
    assert docs[1]['doc']['text'] == 'first doc'
    assert docs[2]['doc']['text'] == 'second doc'

    kb.close()

    # Pairwise scores:
    kb = KB(_DB_PATH, embedding_func)

    records = kb.document_top_pairwise_scores(n = 2)
    assert len(records) == 2

    _, doc_1, doc_2 = records[0]
    assert doc_1['id'] == 1
    assert doc_2['id'] == 2

    _, doc_1, doc_2 = records[1]
    assert doc_1['id'] == 2
    assert doc_2['id'] == 3

    kb.close()

    # Add and retrieve (i.e. test invalidating the embeddings)
    kb = KB(_DB_PATH, embedding_func)

    docs = kb.retrieve('... forth ...', n=1)
    assert len(docs) == 1
    assert docs[0]['doc']['text'] == 'first doc'

    with kb.bulk_add_docs() as add_doc:
        assert add_doc('forth doc') == 4

    docs = kb.retrieve('... forth ...', n=1)
    assert len(docs) == 1
    assert docs[0]['doc']['text'] == 'forth doc'

    kb.close()

    # Graph interface:
    kb = KB(_DB_PATH, embedding_func)

    with kb.bulk_graph_update() as q:
        assert q.count_edges() == 0
        q.add_directed_edge(1, 3, 4)
        q.add_edge(1, 2, 4)
        assert q.count_edges() == 2
        graph = q.build_networkx_graph(multigraph=False)
        assert isinstance(graph, nx.DiGraph)
        q.del_edge(1)
        assert q.count_edges() == 1

    kb.close()

    # Delete and retrieve (i.e. test invalidating the embeddings)
    kb = KB(_DB_PATH, embedding_func)

    docs = kb.retrieve('... forth ...', n=1)
    assert len(docs) == 1
    assert docs[0]['doc']['text'] == 'forth doc'

    with kb.bulk_del_docs() as del_doc:
        del_doc(1)
        del_doc(2)
        del_doc(4)

    docs = kb.retrieve('... forth ...', n=1)
    assert len(docs) == 1
    assert docs[0]['doc']['text'] == 'second doc'

    kb.close()


def test_kb_vector_magnitude():
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
        with kb.bulk_add_docs() as add_doc:
            add_doc("...")
    kb.close()

    # Test magnitude too small:
    kb = KB(_DB_PATH, embedding_func_2)
    with pytest.raises(ValueError):
        with kb.bulk_add_docs() as add_doc:
            add_doc("...")
    kb.close()


def test_kb_keyval_interface():
    # New database!
    kb = KB(_DB_PATH, make_mock_embeddings_func())
    with kb.bulk_keyval_update() as q:
        assert q.count() == 0
        assert not (q.has('a'))
        assert not (q.has('b'))
        with pytest.raises(KeyError):
            q.get('a')
        with pytest.raises(KeyError):
            q.get('b')
        with pytest.raises(ValueError):
            q.get('a', ValueError)
        with pytest.raises(ValueError):
            q.get('b', ValueError)
        assert q.get('a', 'a_default') == 'a_default'
        assert q.get('b', 'b_default') == 'b_default'

        q.set('b', 77)

        assert q.count() == 1
        assert not (q.has('a'))
        assert q.has('b')
        with pytest.raises(KeyError):
            q.get('a')
        assert q.get('b') == 77
        with pytest.raises(ValueError):
            q.get('a', ValueError)
        assert q.get('b', ValueError) == 77
        assert q.get('a', 'a_default') == 'a_default'
        assert q.get('b', 'b_default') == 77

        q.set('b', 'new_val')

        assert q.count() == 1
        assert not (q.has('a'))
        assert q.has('b')
        with pytest.raises(KeyError):
            q.get('a')
        assert q.get('b') == 'new_val'
        with pytest.raises(ValueError):
            q.get('a', ValueError)
        assert q.get('b', ValueError) == 'new_val'
        assert q.get('a', 'a_default') == 'a_default'
        assert q.get('b', 'b_default') == 'new_val'

    kb.close()

    # Prev database; let it remember the embedding function!
    kb = KB(_DB_PATH)
    with kb.bulk_keyval_update() as q:
        assert q.count() == 1
        assert not (q.has('a'))
        assert q.has('b')
        with pytest.raises(KeyError):
            q.get('a')
        assert q.get('b') == 'new_val'
        with pytest.raises(ValueError):
            q.get('a', ValueError)
        assert q.get('b', ValueError) == 'new_val'
        assert q.get('a', 'a_default') == 'a_default'
        assert q.get('b', 'b_default') == 'new_val'

        with pytest.raises(KeyError):
            q.remove('dne')

        q.remove('b')

        assert q.count() == 0
        assert not (q.has('a'))
        assert not (q.has('b'))
        with pytest.raises(KeyError):
            q.get('a')
        with pytest.raises(KeyError):
            q.get('b')
        with pytest.raises(ValueError):
            q.get('a', ValueError)
        with pytest.raises(ValueError):
            q.get('b', ValueError)
        assert q.get('a', 'a_default') == 'a_default'
        assert q.get('b', 'b_default') == 'b_default'

    kb.close()

    # Prev database; let it remember the embedding function!
    kb = KB(_DB_PATH)
    with kb.bulk_keyval_update() as q:
        assert q.count() == 0
        assert not (q.has('a'))
        assert not (q.has('b'))
        with pytest.raises(KeyError):
            q.get('a')
        with pytest.raises(KeyError):
            q.get('b')
        with pytest.raises(ValueError):
            q.get('a', ValueError)
        with pytest.raises(ValueError):
            q.get('b', ValueError)
        assert q.get('a', 'a_default') == 'a_default'
        assert q.get('b', 'b_default') == 'b_default'

        q.set('reason', 'because')
        q.set('answer', 42)
        q.set('age', 87.5)

        assert set(q.items()) == {
            ('reason', 'because'),
            ('answer', 42),
            ('age', 87.5),
        }

        assert set(q) == {
            'reason',
            'answer',
            'age',
        }

        assert 'a' not in q
        assert 'reason' in q

        assert q['reason'] == 'because'
        assert q['answer'] == 42
        with pytest.raises(KeyError):
            q['dne']

        q['a'] = 5
        assert 'a' in q
        assert q['a'] == 5
        del q['a']
        assert 'a' not in q

        assert len(q) == 3

    kb.close()
