import asyncio
from contextlib import contextmanager, asynccontextmanager
import inspect
import sqlite3
import json
import gzip
import shutil
from datetime import datetime, timezone
from threading import Thread
from pathlib import Path

from types import TracebackType
from typing import (
    AsyncIterator, Iterator, List, Tuple, Dict,
    Any, Literal, Optional, Union, Type,
)

import numpy as np

import networkx as nx  # type: ignore

from .embeddings import (
    embedding_to_bytes,
    embedding_from_bytes,
    make_embeddings_func,
    wrap_embeddings_func_check_magnitude,
)

from .types import (
    AsyncDocumentAdder, AsyncDocumentDeleter, AsyncDocumentQuerier,
    AsyncGraphInterface, AsyncKeyValueInterface,
    DocumentAdder, DocumentDeleter, DocumentQuerier,
    GraphInterface, KeyValueInterface,
    DocumentId, DocumentRecord, EdgeId, NetworkXGraphTypes,
    EmbeddingFunc, Retrieval,
)

from .util import (
    chunkify,
    get_top_k,
    get_top_pairs,
    resolve_to_local_uncompressed_file,
    delete_file_if_exists,
)

import logging

_LOG = logging.getLogger(__name__)


_BULK_EMBEDDING_CHUNK_SIZE = 200


# We require the embedding vectors have a magnitude of 1.0 so that we
# can compute the cosine similarly with just a dot product (it's faster that
# way!). Since floating point is... floating point, we'll allow a tolerance.
_EMBEDDING_MAGNITUDE_TOLERANCE = 0.001


assert sqlite3.threadsafety > 0, "sqlite3 was not compiled in thread-safe mode"  # see ref [1]


_SCHEMA_VERSION = 1   # !!! IF YOU CHANGE THE SCHEMA, BUMP THIS VERSION AND WRITE A MIGRATION FUNCTION !!!

_TABLE_DEFS = """

CREATE TABLE IF NOT EXISTS keyval (   -- FOR SVS'S INTERNAL NEEDS
    id INTEGER PRIMARY KEY,
    key TEXT NOT NULL UNIQUE,
    val ANY NOT NULL
) STRICT;

CREATE TABLE IF NOT EXISTS keyval_user (   -- FOR THE USER'S NEEDS
    id INTEGER PRIMARY KEY,
    key TEXT NOT NULL UNIQUE,
    val ANY NOT NULL
) STRICT;

CREATE TABLE IF NOT EXISTS embeddings (
    id INTEGER PRIMARY KEY,
    embedding BLOB NOT NULL
) STRICT;

CREATE TABLE IF NOT EXISTS docs (
    id INTEGER PRIMARY KEY,
    parent_id INTEGER REFERENCES docs(id), -- ALLOW NULL
    level INTEGER NOT NULL,
    text TEXT NOT NULL,
    embedding INTEGER REFERENCES embeddings(id), -- ALLOW NULL
    meta TEXT -- JSON, ALLOW NULL
) STRICT;

CREATE INDEX IF NOT EXISTS idx_docs_parent_id ON docs(parent_id);
CREATE INDEX IF NOT EXISTS idx_docs_level ON docs(level);
CREATE INDEX IF NOT EXISTS idx_docs_embedding ON docs(embedding);

CREATE TABLE IF NOT EXISTS edges (
    id INTEGER PRIMARY KEY,
    a INTEGER REFERENCES docs(id) NOT NULL,  -- first node
    b INTEGER REFERENCES docs(id) NOT NULL,  -- second node
    r INTEGER REFERENCES docs(id) NOT NULL,  -- edge relationship
    w REAL, -- ALLOW NULL                    -- optional weight of this edge
    d INTEGER NOT NULL                       -- 0=undirected (a<->b); 1=directed (a->b)
) STRICT;

CREATE UNIQUE INDEX IF NOT EXISTS idx_edges_abr ON edges(a, b, r);
CREATE INDEX IF NOT EXISTS idx_edges_a ON edges(a);
CREATE INDEX IF NOT EXISTS idx_edges_b ON edges(b);
CREATE INDEX IF NOT EXISTS idx_edges_r ON edges(r);
CREATE INDEX IF NOT EXISTS idx_edges_d ON edges(d);

"""


"""
[1] Thread safety:
SQLite is amazing for *many* reasons, but one is that it correctly handles:
 1. multi-process (multi-connection) to the same database file (https://sqlite.org/faq.html#q5)
 2. multi-threaded access to the sqlite library (https://sqlite.org/faq.html#q6)

However, we're going to play it safer and allow just *one* thread access to each database
connection at once. We'll use asyncio locks around the executor to achieve this, which
should be an easy and lightweight way to do this. I don't expect much (any?) performance
benefit from allowing concurrent access to the underlying SQLite file (despite SQLite
theoretically supporting this) so I'd rather play it safe. We can revisit this if we
change our minds.

Why do I *not* expect perforamcen gain by allowing concurrent access? Well, for one,
SQLite is thread-safe through mutexes, so enough said. Also we're likely bottlnecked
by disk IO anyway.

We *could* still see a performance gain by using prepared queries (not sure yet).
E.g. See https://stackoverflow.com/q/1711631 and https://stackoverflow.com/a/5616969
However, from early testing, I think *Python* will be our bottleneck (not SQLite)
so I haven't tried it yet (again, so, not sure about this yet).
"""


SQLITE_IS_STRICT = True
if sqlite3.sqlite_version_info < (3, 37, 0):
    _LOG.warning("SQLite strict mode not supported; will use non-strict mode")
    _TABLE_DEFS = _TABLE_DEFS.replace(' STRICT;', ';')
    SQLITE_IS_STRICT = False


class _Querier:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def get_key(self, key: str) -> Any:
        res = self.conn.execute(
            """
            SELECT val
            FROM keyval
            WHERE key = ?;
            """,
            (key,),
        )
        row = res.fetchone()
        if row is None:
            raise KeyError(key)
        return row[0]

    def set_key(self, key: str, val: Any) -> None:
        self.conn.execute(
            """
            INSERT INTO keyval (key, val) VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET
                val = excluded.val;
            """,
            (key, val),
        )

    def del_key(self, key: str) -> None:
        res = self.conn.execute(
            """
            DELETE
            FROM keyval
            WHERE key = ?;
            """,
            (key,),
        )
        if res.rowcount == 0:
            raise KeyError(key)

    def get_key_user(self, key: str) -> Any:
        res = self.conn.execute(
            """
            SELECT val
            FROM keyval_user
            WHERE key = ?;
            """,
            (key,),
        )
        row = res.fetchone()
        if row is None:
            raise KeyError(key)
        return row[0]

    def set_key_user(self, key: str, val: Any) -> None:
        self.conn.execute(
            """
            INSERT INTO keyval_user (key, val) VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET
                val = excluded.val;
            """,
            (key, val),
        )

    def del_key_user(self, key: str) -> None:
        res = self.conn.execute(
            """
            DELETE
            FROM keyval_user
            WHERE key = ?;
            """,
            (key,),
        )
        if res.rowcount == 0:
            raise KeyError(key)

    def has_key_user(self, key: str) -> bool:
        res = self.conn.execute(
            """
            SELECT key
            FROM keyval_user
            WHERE key = ?;
            """,
            (key,),
        )
        row = res.fetchone()
        return row is not None

    def count_keys_user(self) -> int:
        res = self.conn.execute(
            """
            SELECT COUNT(*)
            FROM keyval_user;
            """,
            (),
        )
        row = res.fetchone()
        assert row is not None
        n = row[0]
        assert isinstance(n, int)
        return n

    def keyval_user_iterate(self) -> Iterator[Tuple[str, Any]]:
        res = self.conn.execute(
            """
            SELECT key, val
            FROM keyval_user;
            """,
            (),
        )
        for key, val in res:
            yield key, val

    def key_user_iterate(self) -> Iterator[str]:
        res = self.conn.execute(
            """
            SELECT key
            FROM keyval_user;
            """,
            (),
        )
        for key, in res:
            yield key

    def count_docs(self) -> int:
        res = self.conn.execute(
            """
            SELECT COUNT(*)
            FROM docs;
            """,
            (),
        )
        row = res.fetchone()
        assert row is not None
        n = row[0]
        assert isinstance(n, int)
        return n

    def add_doc(
        self,
        text: str,
        parent_id: Optional[DocumentId],
        meta: Optional[Dict[str, Any]],
        embedding: Optional[bytes],
    ) -> DocumentId:
        level: int = 0
        if parent_id is not None:
            res = self.conn.execute(
                """
                SELECT level
                FROM docs
                WHERE id = ?;
                """,
                (parent_id,),
            )
            row = res.fetchone()
            if row is None:
                raise ValueError(f"invalid parent_id: {parent_id}")
            level = row[0] + 1
        emb_id = None
        if embedding is not None:
            res = self.conn.execute(
                """
                INSERT INTO embeddings (embedding)
                VALUES (?);
                """,
                (embedding,),
            )
            assert res.lastrowid is not None
            emb_id = res.lastrowid
        meta_str = None
        if meta is not None:
            meta_str = json.dumps(meta)
        res = self.conn.execute(
            """
            INSERT INTO docs (
                parent_id,
                level,
                text,
                embedding,
                meta
            ) VALUES (
                ?,
                ?,
                ?,
                ?,
                ?
            );
            """,
            (
                parent_id,
                level,
                text,
                emb_id,
                meta_str,
            ),
        )
        assert res.lastrowid is not None
        return res.lastrowid

    def del_doc(self, doc_id: DocumentId) -> None:
        parent_res = self.conn.execute(
            """
            SELECT id
            FROM docs
            WHERE parent_id = ?;
            """,
            (doc_id,),
        )
        if parent_res.fetchone() is not None:
            raise RuntimeError("You cannot delete a document that is a parent.")
        res = self.conn.execute(
            """
            SELECT id
            FROM edges
            WHERE a=? OR b=? OR r=?;
            """,
            (doc_id, doc_id, doc_id),
        )
        edges_to_delete = set([row[0] for row in res])
        for edge_id in edges_to_delete:
            res = self.conn.execute(
                """
                DELETE FROM edges WHERE id = ?;
                """,
                (edge_id,),
            )
            assert res.rowcount == 1
        res = self.conn.execute(
            """
            SELECT embedding
            FROM docs
            WHERE id = ?;
            """,
            (doc_id,),
        )
        row = res.fetchone()
        if row is None:
            raise KeyError(doc_id)
        emb_id = row[0]
        if emb_id is not None:
            res = self.conn.execute(
                """
                DELETE FROM embeddings WHERE id = ?;
                """,
                (emb_id,),
            )
            assert res.rowcount == 1
        res = self.conn.execute(
            """
            DELETE FROM docs WHERE id = ?;
            """,
            (doc_id,),
        )
        assert res.rowcount == 1

    def fetch_doc(
        self,
        doc_id: DocumentId,
        include_embedding: bool,
    ) -> DocumentRecord:
        doc_res = self.conn.execute(
            """
            SELECT
                id,
                parent_id,
                level,
                text,
                embedding,
                meta
            FROM docs
            WHERE id = ?;
            """,
            (doc_id,),
        )
        doc_row = doc_res.fetchone()
        if doc_row is None:
            raise KeyError(doc_id)
        meta = None
        if doc_row[5] is not None:
            meta = json.loads(doc_row[5])
        emb_id = doc_row[4]
        if include_embedding:
            embedding = None
            if emb_id is not None:
                emb_res = self.conn.execute(
                    """
                    SELECT embedding
                    FROM embeddings
                    WHERE id = ?;
                    """,
                    (emb_id,),
                )
                emb_row = emb_res.fetchone()
                if emb_row is None:
                    raise ValueError(f"invalid embedding id: {emb_id}")
                embedding = embedding_from_bytes(emb_row[0])
            return {
                'id': doc_row[0],
                'parent_id': doc_row[1],
                'level': doc_row[2],
                'text': doc_row[3],
                'embedding': embedding,
                'meta': meta,
            }
        else:
            return {
                'id': doc_row[0],
                'parent_id': doc_row[1],
                'level': doc_row[2],
                'text': doc_row[3],
                'embedding': emb_id is not None,
                'meta': meta,
            }

    def fetch_doc_children(
        self,
        doc_id: DocumentId,
        include_embedding: bool,
    ) -> List[DocumentRecord]:
        res = self.conn.execute(
            """
            SELECT id
            FROM docs
            WHERE parent_id = ?;
            """,
            (doc_id,),
        )
        return [
            self.fetch_doc(row[0], include_embedding)
            for row in res
        ]

    def fetch_docs_at_level(
        self,
        level: int,
        include_embedding: bool,
    ) -> List[DocumentRecord]:
        res = self.conn.execute(
            """
            SELECT id
            FROM docs
            WHERE level = ?;
            """,
            (level,),
        )
        return [
            self.fetch_doc(row[0], include_embedding)
            for row in res
        ]

    def fetch_doc_with_emb_id(self, emb_id: int) -> DocumentId:
        res = self.conn.execute(
            """
            SELECT id
            FROM docs
            WHERE embedding = ?;
            """,
            (emb_id,),
        )
        row = res.fetchone()
        if row is None:
            raise KeyError(emb_id)
        doc_id: DocumentId = row[0]
        return doc_id

    def set_doc_embedding(
        self,
        doc_id: DocumentId,
        embedding: Optional[bytes],
        skip_check_old: bool = False,
    ) -> None:
        if not skip_check_old:
            res = self.conn.execute(
                """
                SELECT embedding
                FROM docs
                WHERE id = ?;
                """,
                (doc_id,),
            )
            row = res.fetchone()
            if row is None:
                raise KeyError(doc_id)
            emb_id = row[0]
            if emb_id is not None:
                res = self.conn.execute(
                    """
                    DELETE FROM embeddings WHERE id = ?;
                    """,
                    (emb_id,),
                )
                assert res.rowcount == 1
        emb_id = None
        if embedding is not None:
            res = self.conn.execute(
                """
                INSERT INTO embeddings (embedding)
                VALUES (?);
                """,
                (embedding,),
            )
            assert res.lastrowid is not None
            emb_id = res.lastrowid
        res = self.conn.execute(
            """
            UPDATE docs SET embedding = ? WHERE id = ?;
            """,
            (emb_id, doc_id),
        )
        if res.rowcount != 1:
            raise KeyError(doc_id)

    def build_embeddings_matrix(self) -> Tuple[np.ndarray, np.ndarray]:
        res = self.conn.execute(
            """
            SELECT COUNT(*)
            FROM embeddings;
            """,
            (),
        )
        row = res.fetchone()
        assert row is not None
        n = row[0]
        assert isinstance(n, int)

        res = self.conn.execute(
            """
            SELECT embedding
            FROM embeddings
            LIMIT 1;
            """,
            (),
        )
        row = res.fetchone()
        if row is not None:
            m = len(embedding_from_bytes(row[0]))
        else:
            m = 0

        embeddings_matrix = np.zeros((n, m), dtype=np.float32)
        emb_id_lookup = np.zeros(n, dtype=np.int64)

        res = self.conn.execute(
            """
            SELECT id, embedding
            FROM embeddings;
            """,
            (),
        )
        i = -1
        for i, row in enumerate(res):
            embedding_here = embedding_from_bytes(row[1])
            assert len(embedding_here) == m
            embeddings_matrix[i] = embedding_here
            emb_id_lookup[i] = row[0]
        assert i == n-1

        return embeddings_matrix, emb_id_lookup

    def count_edges(self) -> int:
        res = self.conn.execute(
            """
            SELECT COUNT(*)
            FROM edges;
            """,
            (),
        )
        row = res.fetchone()
        assert row is not None
        n = row[0]
        assert isinstance(n, int)
        return n

    def add_directed_edge(
        self,
        from_doc: DocumentId,
        to_doc: DocumentId,
        relationship: DocumentId,
        weight: Optional[float],
    ) -> EdgeId:
        try:
            res = self.conn.execute(
                """
                INSERT INTO edges (a, b, r, w, d) VALUES (?, ?, ?, ?, 1);
                """,
                (from_doc, to_doc, relationship, weight),
            )
            assert res.lastrowid is not None
            return res.lastrowid
        except sqlite3.IntegrityError:
            raise RuntimeError("This edge triplet already exists!")

    def add_edge(
        self,
        doc1: DocumentId,
        doc2: DocumentId,
        relationship: DocumentId,
        weight: Optional[float],
    ) -> EdgeId:
        try:
            res = self.conn.execute(
                """
                INSERT INTO edges (a, b, r, w, d) VALUES (?, ?, ?, ?, 0);
                """,
                (doc1, doc2, relationship, weight),
            )
            assert res.lastrowid is not None
            return res.lastrowid
        except sqlite3.IntegrityError:
            raise RuntimeError("This edge triplet already exists!")

    def del_edge(self, edge_id: EdgeId) -> None:
        res = self.conn.execute(
            """
            DELETE FROM edges WHERE id = ?;
            """,
            (edge_id,),
        )
        assert res.rowcount == 1

    def build_networkx_graph(
        self,
        multigraph: bool = True,
    ) -> NetworkXGraphTypes:
        res = self.conn.execute(
            """
            SELECT d
            FROM edges
            WHERE d = 1
            LIMIT 1;
            """,
            (),
        )
        row = res.fetchone()
        is_directed_graph = row is not None

        graph = \
            (nx.MultiDiGraph() if is_directed_graph else nx.MultiGraph()) \
            if multigraph else \
            (nx.DiGraph() if is_directed_graph else nx.Graph())

        res = self.conn.execute(
            """
            SELECT a, b, r, w, d
            FROM edges;
            """,
            (),
        )

        for a, b, r, w, d in res:
            attrs = {
                'edge_doc': r,
            }
            if w is not None:
                attrs['weight'] = w
            graph.add_edge(a, b, **attrs)
            if is_directed_graph and d == 0:
                # This is an undirected edge being added to a directed graph,
                # so we'll explicitly add the back-edge.
                graph.add_edge(b, a, **attrs)

        return graph

    def _debug_keyval(self) -> Dict[str, Any]:
        res = self.conn.execute(
            """
            SELECT key, val
            FROM keyval
            """,
            (),
        )
        return {
            row[0]: row[1]
            for row in res
        }

    def _debug_embeddings(self) -> List[Tuple]:
        res = self.conn.execute(
            """
            SELECT *
            FROM embeddings
            """,
            (),
        )
        return [
            tuple(row)
            for row in res
        ]

    def _debug_docs(self) -> List[Tuple]:
        res = self.conn.execute(
            """
            SELECT *
            FROM docs
            """,
            (),
        )
        return [
            tuple(row)
            for row in res
        ]

    def _debug_edges(self) -> List[Tuple]:
        res = self.conn.execute(
            """
            SELECT *
            FROM edges
            """,
            (),
        )
        return [
            tuple(row)
            for row in res
        ]


class _DB:
    def __init__(self, path: Union[Path, str]):
        self.conn: Union[sqlite3.Connection, None] = sqlite3.connect(
            path,
            isolation_level=None,     # <-- manual transactions
            check_same_thread=False,  # <-- see ref [1]
        )
        self.in_transaction = False
        self.path = path
        try:
            self.conn.cursor().executescript(_TABLE_DEFS)
            self.conn.commit()
        except:
            self.conn.close()
            self.conn = None
            raise

    def __enter__(self) -> _Querier:
        assert self.conn is not None
        assert not self.in_transaction
        self.conn.execute('BEGIN TRANSACTION;')
        self.in_transaction = True
        return _Querier(self.conn)

    async def __aenter__(self) -> _Querier:
        return await asyncio.get_running_loop().run_in_executor(None, self.__enter__)

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> Union[Literal[False], None]:
        assert self.conn is not None
        assert self.in_transaction
        if exc_type is not None:
            self.conn.rollback()
            self.in_transaction = False
            _LOG.warning(f'aborting transaction due to exception: {exc_val}')
            assert exc_tb
            return False  # <-- re-raise exception
        else:
            self.conn.commit()
            self.in_transaction = False
            return None

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> Union[Literal[False], None]:
        return await asyncio.get_running_loop().run_in_executor(None, self.__exit__, exc_type, exc_val, exc_tb)

    def vacuum(self) -> None:
        assert self.conn is not None
        assert not self.in_transaction
        self.conn.execute('VACUUM;')

    def close(self) -> None:
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    def check_or_set_schema_version(self) -> None:
        with self as q:
            try:
                schema_version = q.get_key('schema_version')
            except KeyError:
                # This must be a new database, so we'll just set it.
                q.set_key('schema_version', _SCHEMA_VERSION)
                q.set_key('created_datetime', datetime.now(timezone.utc).isoformat())
                return
        if schema_version != _SCHEMA_VERSION:
            # We only have once version so far, so ... how are we here!?
            # PS: In the future, this is where migrations will go.
            raise RuntimeError('unreachable')


class _EmbeddingsMatrix:
    def __init__(self) -> None:
        self.embeddings_matrix: Union[np.ndarray, None] = None
        self.emb_id_lookup: Union[np.ndarray, None] = None

    def invalidate(self) -> None:
        _LOG.info("invalidating cached vectors; they'll be re-built next time you `retrieve()`")
        self.embeddings_matrix = None
        self.emb_id_lookup = None

    def get_sync(self, db: _DB) -> Tuple[np.ndarray, np.ndarray]:
        if self.embeddings_matrix is not None and self.emb_id_lookup is not None:
            _LOG.info("using cached vectors")
            return self.embeddings_matrix, self.emb_id_lookup
        else:
            _LOG.info("re-building cached vectors...")
            with db as q:
                embeddings_matrix, emb_id_lookup = q.build_embeddings_matrix()
            _LOG.info("re-building cached vectors... DONE!")
            self.embeddings_matrix = embeddings_matrix
            self.emb_id_lookup = emb_id_lookup
            return embeddings_matrix, emb_id_lookup

    async def get(self, db: _DB) -> Tuple[np.ndarray, np.ndarray]:
        if self.embeddings_matrix is not None and self.emb_id_lookup is not None:
            _LOG.info("using cached vectors")
            return self.embeddings_matrix, self.emb_id_lookup
        else:
            _LOG.info("re-building cached vectors...")
            def heavy() -> Tuple[np.ndarray, np.ndarray]:
                with db as q:
                    return q.build_embeddings_matrix()
            loop = asyncio.get_running_loop()
            embeddings_matrix, emb_id_lookup = await loop.run_in_executor(None, heavy)
            _LOG.info("re-building cached vectors... DONE!")
            self.embeddings_matrix = embeddings_matrix
            self.emb_id_lookup = emb_id_lookup
            return embeddings_matrix, emb_id_lookup


def _db_check(db: _DB, embedding_func: Optional[EmbeddingFunc]) -> EmbeddingFunc:
    db.check_or_set_schema_version()
    with db as q:
        try:
            db_eparams = json.loads(q.get_key('embedding_func_params'))
        except KeyError:
            db_eparams = None
    init_eparams = getattr(embedding_func, '__embedding_func_params__', None)
    if db_eparams is not None and init_eparams is not None:
        if db_eparams != init_eparams:
            _LOG.warning(f"You are overriding the embedding function stored in the database! Be sure this is what you want to do. Your function: {init_eparams}, database function: {db_eparams}")
        assert embedding_func
    elif db_eparams is not None:
        if embedding_func is not None:
            _LOG.warning(f"You are overriding the embedding function stored in the database! Be sure this is what you want to do. Your function: *unknown params*, database function: {db_eparams}")
        else:
            embedding_func = make_embeddings_func(db_eparams)
    elif init_eparams is not None:
        with db as q:
            q.set_key('embedding_func_params', json.dumps(init_eparams))
        assert embedding_func
    else:
        if embedding_func is not None:
            _LOG.warning("Cannot store this non-standard embeddings function to the database. That's okay, but you'll have to explicitly pass this function to all future instantiations of this database.")
        else:
            raise RuntimeError("No embedding function. You did not passed one to constructor and there is not one in the database. You must pass the embedding function you want to use to the constructor on the *first* usage of a new database; it will be stored in the database for later use.")
    return embedding_func


class AsyncKB:
    """Stupid simple knowledge base."""

    def __init__(
        self,
        local_path_or_remote_url: str,
        embedding_func: Optional[EmbeddingFunc] = None,
        force_fresh_db: bool = False
    ):
        self.local_path_or_remote_url = local_path_or_remote_url
        self.db: Union[_DB, None] = None
        self.db_lock = asyncio.Lock()
        self.embedding_func = embedding_func
        self.embedding_func_orig = embedding_func
        self.embeddings_matrix = _EmbeddingsMatrix()
        self.force_fresh_db = force_fresh_db

    async def _ensure_db(self) -> _DB:
        if self.db is None:
            local_path = await resolve_to_local_uncompressed_file(self.local_path_or_remote_url)
            def heavy() -> _DB:
                if self.force_fresh_db:
                    delete_file_if_exists(local_path)
                db = _DB(local_path)
                try:
                    self.embedding_func = _db_check(db, self.embedding_func)
                    return db
                except:
                    db.close()
                    raise
            db = await asyncio.get_running_loop().run_in_executor(None, heavy)
            self.db = db
        return self.db

    async def load(self) -> None:
        async with self.db_lock:
            db = await self._ensure_db()
            await self.embeddings_matrix.get(db)

    async def close(
        self,
        vacuum: bool = False,
        also_gzip: bool = False,
    ) -> None:
        async with self.db_lock:
            db = await self._ensure_db()
            def heavy() -> Union[Path, str]:
                if vacuum:
                    db.vacuum()
                db.close()
                return db.path
            path = await asyncio.get_running_loop().run_in_executor(None, heavy)
            self.db = None
            self.embedding_func = self.embedding_func_orig
            self.embeddings_matrix.invalidate()
            if also_gzip:
                def heavy2() -> None:
                    _LOG.info(f"AsyncKB.close(): starting gzip...")
                    dest_path = f'{path}.gz'
                    with open(path, 'rb') as from_f:
                        with gzip.open(dest_path, 'wb') as to_f:
                            shutil.copyfileobj(from_f, to_f)
                    _LOG.info(f"AsyncKB.close(): finished gzip: {dest_path}")
                await asyncio.get_running_loop().run_in_executor(None, heavy2)

    async def _get_embedding_func(self) -> EmbeddingFunc:
        assert self.embedding_func   # <-- in all places this is called, the db has been loaded already
        return wrap_embeddings_func_check_magnitude(
            self.embedding_func,
            _EMBEDDING_MAGNITUDE_TOLERANCE,
        )

    async def _get_embeddings_as_bytes(
        self,
        list_of_strings: List[str],
    ) -> List[bytes]:
        func = await self._get_embedding_func()
        list_of_list_of_floats = await func(list_of_strings)
        def heavy() -> List[bytes]:
            return [
                embedding_to_bytes(embedding)
                for embedding in list_of_list_of_floats
            ]
        return await asyncio.get_running_loop().run_in_executor(None, heavy)

    @asynccontextmanager
    async def bulk_add_docs(
        self,
    ) -> AsyncIterator[AsyncDocumentAdder]:
        loop = asyncio.get_running_loop()
        async with self.db_lock:
            db = await self._ensure_db()
            async with db as q:
                in_context_manager = True
                lock = asyncio.Lock()
                needs_embeddings: List[Tuple[DocumentId, str]] = []
                async def add_doc(
                    text: str,
                    parent_id: Optional[DocumentId] = None,
                    meta: Optional[Dict[str, Any]] = None,
                    no_embedding: bool = False,
                ) -> DocumentId:
                    assert in_context_manager, "You may not call this function outside of the context manager!"
                    async with lock:
                        def heavy() -> DocumentId:
                            return q.add_doc(
                                text,
                                parent_id,
                                meta,
                                embedding = None,
                            )
                        doc_id = await loop.run_in_executor(None, heavy)
                        if not no_embedding:
                            needs_embeddings.append((doc_id, text))
                        return doc_id
                try:
                    _LOG.info("starting bulk-add (as new database transaction)")
                    yield add_doc
                finally:
                    in_context_manager = False
                _LOG.info(f"getting {len(needs_embeddings)} document embeddings...")
                for chunk in chunkify(needs_embeddings, _BULK_EMBEDDING_CHUNK_SIZE):
                    doc_ids = [c[0] for c in chunk]
                    texts = [c[1] for c in chunk]
                    embeddings = await self._get_embeddings_as_bytes(texts)
                    def heavy() -> None:
                        for doc_id, embedding in zip(doc_ids, embeddings):
                            q.set_doc_embedding(doc_id, embedding, skip_check_old=True)
                    await loop.run_in_executor(None, heavy)
                _LOG.info(f"*DONE*: got {len(needs_embeddings)} document embeddings")
                self.embeddings_matrix.invalidate()
                _LOG.info("ending bulk-add (committing the database transaction)")

    @asynccontextmanager
    async def bulk_del_docs(
        self,
    ) -> AsyncIterator[AsyncDocumentDeleter]:
        loop = asyncio.get_running_loop()
        async with self.db_lock:
            db = await self._ensure_db()
            async with db as q:
                in_context_manager = True
                lock = asyncio.Lock()
                async def del_doc(doc_id: DocumentId) -> None:
                    assert in_context_manager, "You may not call this function outside of the context manager!"
                    async with lock:
                        def heavy() -> None:
                            return q.del_doc(doc_id)
                        await loop.run_in_executor(None, heavy)
                try:
                    _LOG.info("starting bulk-delete (as new database transaction)")
                    yield del_doc
                finally:
                    in_context_manager = False
                self.embeddings_matrix.invalidate()
                _LOG.info("ending bulk-delete (committing the database transaction)")

    @asynccontextmanager
    async def bulk_query_docs(
        self,
    ) -> AsyncIterator[AsyncDocumentQuerier]:
        loop = asyncio.get_running_loop()
        async with self.db_lock:
            db = await self._ensure_db()
            async with db as q:
                in_context_manager = True
                lock = asyncio.Lock()
                class Querier(AsyncDocumentQuerier):
                    async def count(self) -> int:
                        assert in_context_manager, "You may not call this function outside of the context manager!"
                        async with lock:
                            def heavy() -> int:
                                return q.count_docs()
                            return await loop.run_in_executor(None, heavy)

                    async def query_doc(
                        self,
                        doc_id: DocumentId,
                        include_embedding: bool = False,
                    ) -> DocumentRecord:
                        assert in_context_manager, "You may not call this function outside of the context manager!"
                        async with lock:
                            def heavy() -> DocumentRecord:
                                return q.fetch_doc(doc_id, include_embedding)
                            return await loop.run_in_executor(None, heavy)

                    async def query_children(
                        self,
                        doc_id: DocumentId,
                        include_embedding: bool = False,
                    ) -> List[DocumentRecord]:
                        assert in_context_manager, "You may not call this function outside of the context manager!"
                        async with lock:
                            def heavy() -> List[DocumentRecord]:
                                return q.fetch_doc_children(doc_id, include_embedding)
                            return await loop.run_in_executor(None, heavy)

                    async def query_level(
                        self,
                        level: int,
                        include_embedding: bool = False,
                    ) -> List[DocumentRecord]:
                        assert in_context_manager, "You may not call this function outside of the context manager!"
                        async with lock:
                            def heavy() -> List[DocumentRecord]:
                                return q.fetch_docs_at_level(level, include_embedding)
                            return await loop.run_in_executor(None, heavy)

                    async def dfs_traversal(
                        self,
                        include_embedding: bool = False,
                    ) -> AsyncIterator[DocumentRecord]:
                        async def visit(doc: DocumentRecord) -> AsyncIterator[DocumentRecord]:
                            yield doc
                            children = await self.query_children(doc['id'], include_embedding)
                            for child in children:
                                async for subchild in visit(child):
                                    yield subchild
                        level_0 = await self.query_level(0, include_embedding)
                        for level_0_doc in level_0:
                            async for subdoc in visit(level_0_doc):
                                yield subdoc

                try:
                    yield Querier()
                finally:
                    in_context_manager = False

    async def retrieve(
        self,
        query: str,
        n: int,
    ) -> List[Retrieval]:
        _LOG.info(f"retrieving {n} documents with query string: {query}")
        loop = asyncio.get_running_loop()
        async with self.db_lock:
            db = await self._ensure_db()
            embeddings_matrix, emb_id_lookup = await self.embeddings_matrix.get(db)
        func = await self._get_embedding_func()
        query_vec = np.array((await func([query]))[0], dtype=np.float32)
        _LOG.info("got embedding for query!")
        def superheavy() -> List[Tuple[float, int]]:
            x = np.dot(embeddings_matrix, query_vec)  # numpy go brrr
            emb_ids = []
            for score, index in get_top_k(x, n):
                emb_ids.append((score, int(emb_id_lookup[index])))
            return emb_ids
        emb_ids = await loop.run_in_executor(None, superheavy)
        _LOG.info(f"computed {embeddings_matrix.shape[0]} cosine similarities")
        async with self.db_lock:
            db = await self._ensure_db()
            async with db as q:
                def heavy() -> List[Retrieval]:
                    res: List[Retrieval] = []
                    for score, emb_id in emb_ids:
                        doc_id = q.fetch_doc_with_emb_id(emb_id)
                        doc = q.fetch_doc(doc_id, include_embedding=False)
                        res.append({
                            'score': score,
                            'doc': doc,
                        })
                    _LOG.info(f"retrieved top {n} documents")
                    return res
                return await loop.run_in_executor(None, heavy)

    async def document_top_pairwise_scores(
        self,
        n: int,
    ) -> List[Tuple[float, DocumentRecord, DocumentRecord]]:
        loop = asyncio.get_running_loop()
        async with self.db_lock:
            db = await self._ensure_db()
            embeddings_matrix, emb_id_lookup = await self.embeddings_matrix.get(db)
        n_docs = len(emb_id_lookup)
        _LOG.info(f"computing pairwise similarity over {n_docs} documents")
        def superheavy() -> List[Tuple[float, int, int]]:
            pairwise = np.dot(embeddings_matrix, embeddings_matrix.T)
            return [
                (score, int(emb_id_lookup[i1]), int(emb_id_lookup[i2]))
                for score, i1, i2 in get_top_pairs(pairwise, n)
            ]
        pairwise_scores = await loop.run_in_executor(None, superheavy)
        _LOG.info(f"computed {n_docs * n_docs} pairwise cosine similarities")
        async with self.db_lock:
            db = await self._ensure_db()
            async with db as q:
                def heavy() -> List[Tuple[float, DocumentRecord, DocumentRecord]]:
                    emb_id_to_doc_id: Dict[int, DocumentId] = {}
                    for emb_id in set(emb_id for _, emb_id_1, emb_id_2 in pairwise_scores for emb_id in (emb_id_1, emb_id_2)):
                        emb_id_to_doc_id[emb_id] = q.fetch_doc_with_emb_id(emb_id)
                    doc_lookup: Dict[DocumentId, DocumentRecord] = {}
                    for doc_id in emb_id_to_doc_id.values():
                        doc_lookup[doc_id] = q.fetch_doc(doc_id, include_embedding=False)
                    res: List[Tuple[float, DocumentRecord, DocumentRecord]] = []
                    for score, emb_id_1, emb_id_2 in pairwise_scores:
                        doc_1 = doc_lookup[emb_id_to_doc_id[emb_id_1]]
                        doc_2 = doc_lookup[emb_id_to_doc_id[emb_id_2]]
                        res.append((score, doc_1, doc_2))
                    _LOG.info(f"retrieved top {n} document pairs")
                    return res
                return await loop.run_in_executor(None, heavy)

    @asynccontextmanager
    async def bulk_graph_update(
        self,
    ) -> AsyncIterator[AsyncGraphInterface]:
        loop = asyncio.get_running_loop()
        async with self.db_lock:
            db = await self._ensure_db()
            async with db as q:
                in_context_manager = True
                lock = asyncio.Lock()
                class Querier(AsyncGraphInterface):
                    async def count_edges(self) -> int:
                        assert in_context_manager, "You may not call this function outside of the context manager!"
                        async with lock:
                            def heavy() -> int:
                                return q.count_edges()
                            return await loop.run_in_executor(None, heavy)

                    async def add_directed_edge(
                        self,
                        from_doc: DocumentId,
                        to_doc: DocumentId,
                        relationship: DocumentId,
                        weight: Optional[float] = None,
                    ) -> EdgeId:
                        assert in_context_manager, "You may not call this function outside of the context manager!"
                        async with lock:
                            def heavy() -> EdgeId:
                                return q.add_directed_edge(
                                    from_doc,
                                    to_doc,
                                    relationship,
                                    weight,
                                )
                            return await loop.run_in_executor(None, heavy)

                    async def add_edge(
                        self,
                        doc1: DocumentId,
                        doc2: DocumentId,
                        relationship: DocumentId,
                        weight: Optional[float] = None,
                    ) -> EdgeId:
                        assert in_context_manager, "You may not call this function outside of the context manager!"
                        async with lock:
                            def heavy() -> EdgeId:
                                return q.add_edge(
                                    doc1,
                                    doc2,
                                    relationship,
                                    weight,
                                )
                            return await loop.run_in_executor(None, heavy)

                    async def del_edge(self, edge_id: EdgeId) -> None:
                        assert in_context_manager, "You may not call this function outside of the context manager!"
                        async with lock:
                            def heavy() -> None:
                                return q.del_edge(edge_id)
                            return await loop.run_in_executor(None, heavy)

                    async def build_networkx_graph(
                        self,
                        multigraph: bool = True,
                    ) -> NetworkXGraphTypes:
                        assert in_context_manager, "You may not call this function outside of the context manager!"
                        async with lock:
                            def heavy() -> NetworkXGraphTypes:
                                return q.build_networkx_graph(multigraph)
                            return await loop.run_in_executor(None, heavy)

                try:
                    yield Querier()
                finally:
                    in_context_manager = False

    @asynccontextmanager
    async def bulk_keyval_update(
        self,
    ) -> AsyncIterator[AsyncKeyValueInterface]:
        loop = asyncio.get_running_loop()
        async with self.db_lock:
            db = await self._ensure_db()
            async with db as q:
                in_context_manager = True
                lock = asyncio.Lock()
                class Querier(AsyncKeyValueInterface):
                    async def has(self, key: str) -> bool:
                        assert in_context_manager, "You may not call this function outside of the context manager!"
                        async with lock:
                            def heavy() -> bool:
                                return q.has_key_user(key)
                            return await loop.run_in_executor(None, heavy)

                    async def get(self, key: str, default: Any = KeyError) -> Any:
                        assert in_context_manager, "You may not call this function outside of the context manager!"
                        async with lock:
                            def heavy() -> Any:
                                try:
                                    return q.get_key_user(key)
                                except KeyError:
                                    if inspect.isclass(default):
                                        if issubclass(default, KeyError):
                                            raise
                                        elif issubclass(default, Exception):
                                            raise default()
                                    return default
                            return await loop.run_in_executor(None, heavy)

                    async def set(self, key: str, val: Any) -> None:
                        assert in_context_manager, "You may not call this function outside of the context manager!"
                        async with lock:
                            def heavy() -> None:
                                return q.set_key_user(key, val)
                            return await loop.run_in_executor(None, heavy)

                    async def remove(self, key: str) -> None:
                        assert in_context_manager, "You may not call this function outside of the context manager!"
                        async with lock:
                            def heavy() -> None:
                                return q.del_key_user(key)
                            return await loop.run_in_executor(None, heavy)

                    async def count(self) -> int:
                        assert in_context_manager, "You may not call this function outside of the context manager!"
                        async with lock:
                            def heavy() -> int:
                                return q.count_keys_user()
                            return await loop.run_in_executor(None, heavy)

                    async def items(self) -> AsyncIterator[Tuple[str, Any]]:
                        assert in_context_manager, "You may not call this function outside of the context manager!"
                        async with lock:
                            queue: asyncio.Queue[Union[Tuple[str, Any], None, Exception]] = asyncio.Queue()
                            def heavy() -> None:
                                try:
                                    for item in q.keyval_user_iterate():
                                        loop.call_soon_threadsafe(queue.put_nowait, item)
                                    loop.call_soon_threadsafe(queue.put_nowait, None)
                                except Exception as e:
                                    loop.call_soon_threadsafe(queue.put_nowait, e)
                            task = loop.run_in_executor(None, heavy)
                            while True:
                                item = await queue.get()
                                if item is None:
                                    break
                                elif isinstance(item, Exception):
                                    raise item
                                yield item
                            await task

                try:
                    yield Querier()
                finally:
                    in_context_manager = False


def _loop_main(loop: asyncio.AbstractEventLoop) -> None:
    asyncio.set_event_loop(loop)
    loop.run_forever()


class KB:
    """Stupid simple knowledge base."""

    def __init__(
        self,
        local_path_or_remote_url: str,
        embedding_func: Optional[EmbeddingFunc] = None,
        force_fresh_db: bool = False
    ):
        self.local_path_or_remote_url = local_path_or_remote_url
        self.db: Union[_DB, None] = None
        self.embedding_func = embedding_func
        self.embedding_func_orig = embedding_func
        self.embeddings_matrix = _EmbeddingsMatrix()

        self.loop = asyncio.new_event_loop()
        self.thread: Union[Thread, None] = Thread(target=_loop_main, args=(self.loop,))
        self.thread.daemon = True
        self.thread.start()

        local_path = asyncio.run_coroutine_threadsafe(resolve_to_local_uncompressed_file(self.local_path_or_remote_url), self.loop).result()
        if force_fresh_db:
            delete_file_if_exists(local_path)
        self.db = _DB(local_path)
        try:
            self.embedding_func = _db_check(self.db, self.embedding_func)
        except:
            self.close()
            raise

    def close(
        self,
        vacuum: bool = False,
        also_gzip: bool = False,
    ) -> None:
        if self.thread is not None:
            async def _stop() -> None:
                self.loop.stop()
            asyncio.run_coroutine_threadsafe(_stop(), self.loop)
            self.thread.join()
            self.thread = None
        if self.db is not None:
            if vacuum:
                self.db.vacuum()
            self.db.close()
            path = self.db.path
            self.db = None
            self.embedding_func = self.embedding_func_orig
            self.embeddings_matrix.invalidate()
            if also_gzip:
                _LOG.info(f"KB.close(): starting gzip...")
                dest_path = f'{path}.gz'
                with open(path, 'rb') as from_f:
                    with gzip.open(dest_path, 'wb') as to_f:
                        shutil.copyfileobj(from_f, to_f)
                _LOG.info(f"KB.close(): finished gzip: {dest_path}")

    def _get_embedding_func(self) -> EmbeddingFunc:
        assert self.embedding_func   # <-- true if we haven't closed
        return wrap_embeddings_func_check_magnitude(
            self.embedding_func,
            _EMBEDDING_MAGNITUDE_TOLERANCE,
        )

    def _get_embeddings_as_bytes(
        self,
        list_of_strings: List[str],
    ) -> List[bytes]:
        func = self._get_embedding_func()
        list_of_list_of_floats = asyncio.run_coroutine_threadsafe(func(list_of_strings), self.loop).result()
        return [
            embedding_to_bytes(embedding)
            for embedding in list_of_list_of_floats
        ]

    @contextmanager
    def bulk_add_docs(
        self,
    ) -> Iterator[DocumentAdder]:
        assert self.db is not None
        with self.db as q:
            in_context_manager = True
            needs_embeddings: List[Tuple[DocumentId, str]] = []
            def add_doc(
                text: str,
                parent_id: Optional[DocumentId] = None,
                meta: Optional[Dict[str, Any]] = None,
                no_embedding: bool = False,
            ) -> DocumentId:
                assert in_context_manager, "You may not call this function outside of the context manager!"
                doc_id = q.add_doc(
                    text,
                    parent_id,
                    meta,
                    embedding = None,
                )
                if not no_embedding:
                    needs_embeddings.append((doc_id, text))
                return doc_id
            try:
                _LOG.info("starting bulk-add (as new database transaction)")
                yield add_doc
            finally:
                in_context_manager = False
            _LOG.info(f"getting {len(needs_embeddings)} document embeddings...")
            for chunk in chunkify(needs_embeddings, _BULK_EMBEDDING_CHUNK_SIZE):
                doc_ids = [c[0] for c in chunk]
                texts = [c[1] for c in chunk]
                embeddings = self._get_embeddings_as_bytes(texts)
                for doc_id, embedding in zip(doc_ids, embeddings):
                    q.set_doc_embedding(doc_id, embedding, skip_check_old=True)
            _LOG.info(f"*DONE*: got {len(needs_embeddings)} document embeddings")
            self.embeddings_matrix.invalidate()
            _LOG.info("ending bulk-add (committing the database transaction)")

    @contextmanager
    def bulk_del_docs(
        self,
    ) -> Iterator[DocumentDeleter]:
        assert self.db is not None
        with self.db as q:
            in_context_manager = True
            def del_doc(doc_id: DocumentId) -> None:
                assert in_context_manager, "You may not call this function outside of the context manager!"
                return q.del_doc(doc_id)
            try:
                _LOG.info("starting bulk-delete (as new database transaction)")
                yield del_doc
            finally:
                in_context_manager = False
            self.embeddings_matrix.invalidate()
            _LOG.info("ending bulk-delete (committing the database transaction)")

    @contextmanager
    def bulk_query_docs(
        self,
    ) -> Iterator[DocumentQuerier]:
        assert self.db is not None
        with self.db as q:
            in_context_manager = True
            class Querier(DocumentQuerier):
                def count(self) -> int:
                    assert in_context_manager, "You may not call this function outside of the context manager!"
                    return q.count_docs()

                def query_doc(
                    self,
                    doc_id: DocumentId,
                    include_embedding: bool = False,
                ) -> DocumentRecord:
                    assert in_context_manager, "You may not call this function outside of the context manager!"
                    return q.fetch_doc(doc_id, include_embedding)

                def query_children(
                    self,
                    doc_id: DocumentId,
                    include_embedding: bool = False,
                ) -> List[DocumentRecord]:
                    assert in_context_manager, "You may not call this function outside of the context manager!"
                    return q.fetch_doc_children(doc_id, include_embedding)

                def query_level(
                    self,
                    level: int,
                    include_embedding: bool = False,
                ) -> List[DocumentRecord]:
                    assert in_context_manager, "You may not call this function outside of the context manager!"
                    return q.fetch_docs_at_level(level, include_embedding)

                def dfs_traversal(
                    self,
                    include_embedding: bool = False,
                ) -> Iterator[DocumentRecord]:
                    def visit(doc: DocumentRecord) -> Iterator[DocumentRecord]:
                        yield doc
                        children = self.query_children(doc['id'], include_embedding)
                        for child in children:
                            for subchild in visit(child):
                                yield subchild
                    level_0 = self.query_level(0, include_embedding)
                    for level_0_doc in level_0:
                        for subdoc in visit(level_0_doc):
                            yield subdoc

            try:
                yield Querier()
            finally:
                in_context_manager = False

    def retrieve(
        self,
        query: str,
        n: int,
    ) -> List[Retrieval]:
        _LOG.info(f"retrieving {n} documents with query string: {query}")
        assert self.db is not None
        embeddings_matrix, emb_id_lookup = self.embeddings_matrix.get_sync(self.db)
        func = self._get_embedding_func()
        query_list_floats = asyncio.run_coroutine_threadsafe(func([query]), self.loop).result()[0]
        query_vec = np.array(query_list_floats, dtype=np.float32)
        _LOG.info("got embedding for query!")
        def superheavy() -> List[Tuple[float, int]]:
            x = np.dot(embeddings_matrix, query_vec)  # numpy go brrr
            emb_ids = []
            for score, index in get_top_k(x, n):
                emb_ids.append((score, int(emb_id_lookup[index])))
            return emb_ids
        emb_ids = superheavy()
        _LOG.info(f"computed {embeddings_matrix.shape[0]} cosine similarities")
        with self.db as q:
            res: List[Retrieval] = []
            for score, emb_id in emb_ids:
                doc_id = q.fetch_doc_with_emb_id(emb_id)
                doc = q.fetch_doc(doc_id, include_embedding=False)
                res.append({
                    'score': score,
                    'doc': doc,
                })
            _LOG.info(f"retrieved top {n} documents")
            return res

    def document_top_pairwise_scores(
        self,
        n: int,
    ) -> List[Tuple[float, DocumentRecord, DocumentRecord]]:
        assert self.db is not None
        embeddings_matrix, emb_id_lookup = self.embeddings_matrix.get_sync(self.db)
        n_docs = len(emb_id_lookup)
        _LOG.info(f"computing pairwise similarity over {n_docs} documents")
        def superheavy() -> List[Tuple[float, int, int]]:
            pairwise = np.dot(embeddings_matrix, embeddings_matrix.T)
            return [
                (score, int(emb_id_lookup[i1]), int(emb_id_lookup[i2]))
                for score, i1, i2 in get_top_pairs(pairwise, n)
            ]
        pairwise_scores = superheavy()
        _LOG.info(f"computed {n_docs * n_docs} pairwise cosine similarities")
        with self.db as q:
            emb_id_to_doc_id: Dict[int, DocumentId] = {}
            for emb_id in set(emb_id for _, emb_id_1, emb_id_2 in pairwise_scores for emb_id in (emb_id_1, emb_id_2)):
                emb_id_to_doc_id[emb_id] = q.fetch_doc_with_emb_id(emb_id)
            doc_lookup: Dict[DocumentId, DocumentRecord] = {}
            for doc_id in emb_id_to_doc_id.values():
                doc_lookup[doc_id] = q.fetch_doc(doc_id, include_embedding=False)
            res: List[Tuple[float, DocumentRecord, DocumentRecord]] = []
            for score, emb_id_1, emb_id_2 in pairwise_scores:
                doc_1 = doc_lookup[emb_id_to_doc_id[emb_id_1]]
                doc_2 = doc_lookup[emb_id_to_doc_id[emb_id_2]]
                res.append((score, doc_1, doc_2))
            _LOG.info(f"retrieved top {n} document pairs")
            return res

    @contextmanager
    def bulk_graph_update(
        self,
    ) -> Iterator[GraphInterface]:
        assert self.db is not None
        with self.db as q:
            in_context_manager = True
            class Querier(GraphInterface):
                def count_edges(self) -> int:
                    assert in_context_manager, "You may not call this function outside of the context manager!"
                    return q.count_edges()

                def add_directed_edge(
                    self,
                    from_doc: DocumentId,
                    to_doc: DocumentId,
                    relationship: DocumentId,
                    weight: Optional[float] = None,
                ) -> EdgeId:
                    assert in_context_manager, "You may not call this function outside of the context manager!"
                    return q.add_directed_edge(
                        from_doc,
                        to_doc,
                        relationship,
                        weight,
                    )

                def add_edge(
                    self,
                    doc1: DocumentId,
                    doc2: DocumentId,
                    relationship: DocumentId,
                    weight: Optional[float] = None,
                ) -> EdgeId:
                    assert in_context_manager, "You may not call this function outside of the context manager!"
                    return q.add_edge(
                        doc1,
                        doc2,
                        relationship,
                        weight,
                    )

                def del_edge(self, edge_id: EdgeId) -> None:
                    assert in_context_manager, "You may not call this function outside of the context manager!"
                    return q.del_edge(edge_id)

                def build_networkx_graph(
                    self,
                    multigraph: bool = True,
                ) -> NetworkXGraphTypes:
                    assert in_context_manager, "You may not call this function outside of the context manager!"
                    return q.build_networkx_graph(multigraph)

            try:
                yield Querier()
            finally:
                in_context_manager = False

    @contextmanager
    def bulk_keyval_update(
        self,
    ) -> Iterator[KeyValueInterface]:
        assert self.db is not None
        with self.db as q:
            in_context_manager = True
            class Querier(KeyValueInterface):
                def has(self, key: str) -> bool:
                    assert in_context_manager, "You may not call this function outside of the context manager!"
                    return q.has_key_user(key)

                def __contains__(self, key: str) -> bool:
                    return self.has(key)

                def get(self, key: str, default: Any = KeyError) -> Any:
                    assert in_context_manager, "You may not call this function outside of the context manager!"
                    try:
                        return q.get_key_user(key)
                    except KeyError:
                        if inspect.isclass(default):
                            if issubclass(default, KeyError):
                                raise
                            elif issubclass(default, Exception):
                                raise default()
                        return default

                def __getitem__(self, key: str) -> Any:
                    return self.get(key)

                def set(self, key: str, val: Any) -> None:
                    assert in_context_manager, "You may not call this function outside of the context manager!"
                    return q.set_key_user(key, val)

                def __setitem__(self, key: str, val: Any) -> None:
                    return self.set(key, val)

                def remove(self, key: str) -> None:
                    assert in_context_manager, "You may not call this function outside of the context manager!"
                    return q.del_key_user(key)

                def __delitem__(self, key: str) -> None:
                    return self.remove(key)

                def count(self) -> int:
                    assert in_context_manager, "You may not call this function outside of the context manager!"
                    return q.count_keys_user()

                def __len__(self) -> int:
                    return self.count()

                def items(self) -> Iterator[Tuple[str, Any]]:
                    assert in_context_manager, "You may not call this function outside of the context manager!"
                    for item in q.keyval_user_iterate():
                        yield item

                def __iter__(self) -> Iterator[str]:
                    assert in_context_manager, "You may not call this function outside of the context manager!"
                    for key in q.key_user_iterate():
                        yield key

            try:
                yield Querier()
            finally:
                in_context_manager = False

    def __len__(self) -> int:
        with self.bulk_query_docs() as q:
            return q.count()
