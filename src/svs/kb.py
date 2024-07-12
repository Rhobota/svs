import asyncio
from contextlib import contextmanager, asynccontextmanager
import sqlite3
import json
from datetime import datetime, timezone
from threading import Thread

from types import TracebackType
from typing import (
    AsyncIterator, Iterator, List, Tuple, Dict,
    Any, Literal, Optional, Union, Type,
)

import numpy as np

from .embeddings import (
    embedding_to_bytes,
    embedding_from_bytes,
    make_embeddings_func,
    wrap_embeddings_func_check_magnitude,
)

from .types import (
    AsyncDocumentAdder, AsyncDocumentDeleter, AsyncDocumentQuerier,
    DocumentAdder, DocumentDeleter, DocumentQuerier,
    DocumentId, DocumentRecord,
    EmbeddingFunc, Retrieval,
)

from .util import chunkify, get_top_k

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

CREATE TABLE IF NOT EXISTS keyval (
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


class _DB:
    def __init__(self, path: str):
        self.conn: Union[sqlite3.Connection, None] = sqlite3.connect(
            path,
            isolation_level=None,     # <-- manual transactions
            check_same_thread=False,  # <-- see ref [1]
        )
        self.in_transaction = False
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
        self.embeddings_matrix = None
        self.emb_id_lookup = None

    def get_sync(self, db: _DB) -> Tuple[np.ndarray, np.ndarray]:
        if self.embeddings_matrix is not None and self.emb_id_lookup is not None:
            return self.embeddings_matrix, self.emb_id_lookup
        else:
            with db as q:
                embeddings_matrix, emb_id_lookup = q.build_embeddings_matrix()
            self.embeddings_matrix = embeddings_matrix
            self.emb_id_lookup = emb_id_lookup
            return embeddings_matrix, emb_id_lookup

    async def get(self, db: _DB) -> Tuple[np.ndarray, np.ndarray]:
        if self.embeddings_matrix is not None and self.emb_id_lookup is not None:
            return self.embeddings_matrix, self.emb_id_lookup
        else:
            def heavy() -> Tuple[np.ndarray, np.ndarray]:
                with db as q:
                    return q.build_embeddings_matrix()
            loop = asyncio.get_running_loop()
            embeddings_matrix, emb_id_lookup = await loop.run_in_executor(None, heavy)
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
        db_file_path: str,
        embedding_func: Optional[EmbeddingFunc] = None,
    ):
        self.db_file_path = db_file_path
        self.db: Union[_DB, None] = None
        self.db_lock = asyncio.Lock()
        self.embedding_func = embedding_func
        self.embedding_func_orig = embedding_func
        self.embeddings_matrix = _EmbeddingsMatrix()

    async def _ensure_db(self) -> _DB:
        if self.db is None:
            def heavy() -> _DB:
                db = _DB(self.db_file_path)
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

    async def close(self, vacuum: bool = False) -> None:
        async with self.db_lock:
            db = await self._ensure_db()
            def heavy() -> None:
                if vacuum:
                    db.vacuum()
                db.close()
            await asyncio.get_running_loop().run_in_executor(None, heavy)
            self.db = None
            self.embedding_func = self.embedding_func_orig
            self.embeddings_matrix.invalidate()

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
                    yield add_doc
                finally:
                    in_context_manager = False
                for chunk in chunkify(needs_embeddings, _BULK_EMBEDDING_CHUNK_SIZE):
                    doc_ids = [c[0] for c in chunk]
                    texts = [c[1] for c in chunk]
                    embeddings = await self._get_embeddings_as_bytes(texts)
                    def heavy() -> None:
                        for doc_id, embedding in zip(doc_ids, embeddings):
                            q.set_doc_embedding(doc_id, embedding, skip_check_old=True)
                    await loop.run_in_executor(None, heavy)
                self.embeddings_matrix.invalidate()

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
                    yield del_doc
                finally:
                    in_context_manager = False
                self.embeddings_matrix.invalidate()

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
        include_documents: bool = True,
    ) -> List[Retrieval]:
        loop = asyncio.get_running_loop()
        async with self.db_lock:
            db = await self._ensure_db()
            embeddings_matrix, emb_id_lookup = await self.embeddings_matrix.get(db)
        func = await self._get_embedding_func()
        query_vec = np.array((await func([query]))[0], dtype=np.float32)
        def superheavy() -> List[Tuple[float, int]]:
            x = np.dot(embeddings_matrix, query_vec)  # numpy go brrr
            emb_ids = []
            for score, index in get_top_k(x, n):
                emb_ids.append((score, int(emb_id_lookup[index])))
            return emb_ids
        emb_ids = await loop.run_in_executor(None, superheavy)
        async with self.db_lock:
            db = await self._ensure_db()
            async with db as q:
                def heavy() -> List[Retrieval]:
                    res: List[Retrieval] = []
                    for score, emb_id in emb_ids:
                        doc_id = q.fetch_doc_with_emb_id(emb_id)
                        doc = q.fetch_doc(doc_id, include_embedding=False) if include_documents else None
                        res.append({
                            'score': score,
                            'doc_id': doc_id,
                            'doc': doc,
                        })
                    return res
                return await loop.run_in_executor(None, heavy)


def _loop_main(loop: asyncio.AbstractEventLoop) -> None:
    asyncio.set_event_loop(loop)
    loop.run_forever()


class KB:
    """Stupid simple knowledge base."""

    def __init__(
        self,
        db_file_path: str,
        embedding_func: Optional[EmbeddingFunc] = None,
    ):
        self.db_file_path = db_file_path
        self.db: Union[_DB, None] = _DB(db_file_path)
        self.embedding_func = embedding_func
        self.embedding_func_orig = embedding_func
        self.embeddings_matrix = _EmbeddingsMatrix()
        try:
            self.embedding_func = _db_check(self.db, self.embedding_func)
        except:
            self.db.close()
            self.db = None
            raise
        self.loop = asyncio.new_event_loop()
        self.thread: Union[Thread, None] = Thread(target=_loop_main, args=(self.loop,))
        self.thread.daemon = True
        self.thread.start()

    def close(self, vacuum: bool = False) -> None:
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
            self.db = None
            self.embedding_func = self.embedding_func_orig
            self.embeddings_matrix.invalidate()

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
                yield add_doc
            finally:
                in_context_manager = False
            for chunk in chunkify(needs_embeddings, _BULK_EMBEDDING_CHUNK_SIZE):
                doc_ids = [c[0] for c in chunk]
                texts = [c[1] for c in chunk]
                embeddings = self._get_embeddings_as_bytes(texts)
                for doc_id, embedding in zip(doc_ids, embeddings):
                    q.set_doc_embedding(doc_id, embedding, skip_check_old=True)
            self.embeddings_matrix.invalidate()

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
                yield del_doc
            finally:
                in_context_manager = False
            self.embeddings_matrix.invalidate()

    @contextmanager
    def bulk_query_docs(
        self,
    ) -> Iterator[DocumentQuerier]:
        assert self.db is not None
        with self.db as q:
            in_context_manager = True
            class Querier(DocumentQuerier):
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
        include_documents: bool = True,
    ) -> List[Retrieval]:
        assert self.db is not None
        embeddings_matrix, emb_id_lookup = self.embeddings_matrix.get_sync(self.db)
        func = self._get_embedding_func()
        query_list_floats = asyncio.run_coroutine_threadsafe(func([query]), self.loop).result()[0]
        query_vec = np.array(query_list_floats, dtype=np.float32)
        def superheavy() -> List[Tuple[float, int]]:
            x = np.dot(embeddings_matrix, query_vec)  # numpy go brrr
            emb_ids = []
            for score, index in get_top_k(x, n):
                emb_ids.append((score, int(emb_id_lookup[index])))
            return emb_ids
        emb_ids = superheavy()
        with self.db as q:
            res: List[Retrieval] = []
            for score, emb_id in emb_ids:
                doc_id = q.fetch_doc_with_emb_id(emb_id)
                doc = q.fetch_doc(doc_id, include_embedding=False) if include_documents else None
                res.append({
                    'score': score,
                    'doc_id': doc_id,
                    'doc': doc,
                })
            return res
