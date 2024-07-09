import asyncio
import sqlite3
import json

from types import TracebackType
from typing import (
    List, Tuple, Dict, Any, Literal,
    Optional, Union, Type, cast,
)

import numpy as np

from .embeddings import (
    embedding_to_bytes,
    embedding_from_bytes,
    make_embeddings_func,
)

from .types import DocumentId, DocumentRecord, EmbeddingFunc

import logging

_LOG = logging.getLogger(__name__)


assert sqlite3.threadsafety > 0, "sqlite3 was not compiled in thread-safe mode"  # see ref [1]


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
benefit from allowing concurrent reads to the underlying SQLite file (despite SQLite
theoretically supporting this) so I'd rather play it safe. We can revisit this if we
change our minds.
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

    def fetch_doc(self, doc_id: DocumentId) -> DocumentRecord:
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
        emb_id = doc_row[4]
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
        meta = None
        if doc_row[5] is not None:
            meta = json.loads(doc_row[5])
        return {
            'id': doc_row[0],
            'parent_id': doc_row[1],
            'level': doc_row[2],
            'text': doc_row[3],
            'embedding': embedding,
            'meta': meta,
        }

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

    def vacuum(self) -> None:
        assert self.conn is not None
        assert not self.in_transaction
        self.conn.execute('VACUUM;')

    def close(self) -> None:
        if self.conn is not None:
            self.conn.close()
            self.conn = None


class KB:
    """Stupid simple knowledge base."""

    def __init__(
        self,
        db_file_path: str,
        embedding_func: Optional[EmbeddingFunc] = None,
    ):
        self.loop = asyncio.get_running_loop()
        self.db_file_path = db_file_path
        self.db: Union[_DB, None] = None
        self.db_lock = asyncio.Lock()
        self.embedding_func = embedding_func

    async def _ensure_db(self) -> _DB:
        if self.db is None:
            def heavy() -> _DB:
                db = _DB(self.db_file_path)
                try:
                    with db as q:
                        try:
                            db_eparams = json.loads(q.get_key('embedding_func_params'))
                        except KeyError:
                            db_eparams = None
                    init_eparams = getattr(self.embedding_func, '__embedding_func_params__', None)
                    if db_eparams is not None and init_eparams is not None:
                        if db_eparams != init_eparams:
                            _LOG.warning(f"You are overriding the embedding function stored in the database! Be sure this is what you want to do. Your function: {init_eparams}, database function: {db_eparams}")
                    elif db_eparams is not None:
                        if self.embedding_func is not None:
                            _LOG.warning(f"You are overriding the embedding function stored in the database! Be sure this is what you want to do. Your function: *unknown params*, database function: {db_eparams}")
                        else:
                            self.embedding_func = make_embeddings_func(db_eparams)
                    elif init_eparams is not None:
                        with db as q:
                            q.set_key('embedding_func_params', json.dumps(init_eparams))
                    else:
                        raise RuntimeError("No embedding function. You did not passed one to constructor and there is not one in the database. You must pass the embedding function you want to use to the constructor on the *first* usage of a new database; it will be stored in the database for later use.")
                    return db
                except:
                    db.close()
                    raise
            db = await self.loop.run_in_executor(None, heavy)
            self.db = db
        return self.db

    async def close(self, vacuum: bool = False) -> None:
        async with self.db_lock:
            db = await self._ensure_db()
            def heavy() -> None:
                if vacuum:
                    db.vacuum()
                db.close()
            await self.loop.run_in_executor(None, heavy)
            self.db = None

    async def _get_embedding_func(self) -> EmbeddingFunc:
        if self.embedding_func is None:
            async with self.db_lock:
                # Loading the database will load the embedding func.
                await self._ensure_db()
                assert self.embedding_func
        return self.embedding_func

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
        return await self.loop.run_in_executor(None, heavy)

    async def add_doc(
        self,
        text: str,
        parent_id: Optional[DocumentId] = None,
        meta: Optional[Dict[str, Any]] = None,
        no_embedding: bool = False,
    ) -> DocumentId:
        embedding = None
        if not no_embedding:
            embedding = (await self._get_embeddings_as_bytes([text]))[0]
        async with self.db_lock:
            db = await self._ensure_db()
            def heavy() -> DocumentId:
                with db as q:
                    return q.add_doc(
                        text,
                        parent_id,
                        meta,
                        embedding,
                    )
            return await self.loop.run_in_executor(None, heavy)

    async def del_doc(self, doc_id: DocumentId) -> None:
        async with self.db_lock:
            db = await self._ensure_db()
            def heavy() -> None:
                with db as q:
                    return q.del_doc(doc_id)
            return await self.loop.run_in_executor(None, heavy)

    async def retrieve(
        self,
        query: str,
        n: int,
    ) -> List[DocumentRecord]:
        return []   # TODO

    async def set_key(self, key: str, val: Any) -> None:
        db = await self._ensure_db()
        def heavy() -> None:
            with db as q:
                return q.set_key(key, val)
        async with self.db_lock:
            return await self.loop.run_in_executor(None, heavy)

    async def del_key(self, key: str) -> None:
        db = await self._ensure_db()
        def heavy() -> None:
            with db as q:
                return q.del_key(key)
        async with self.db_lock:
            return await self.loop.run_in_executor(None, heavy)
