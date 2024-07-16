from typing import (
    Any, Awaitable, Callable, Dict, List,
    Optional, Protocol, TypedDict, Union,
    AsyncIterator, Iterator,
)

import abc


EmbeddingFunc = Callable[[List[str]], Awaitable[List[List[float]]]]


DocumentId = int


class DocumentRecord(TypedDict):
    id: DocumentId
    parent_id: Optional[DocumentId]
    level: int
    text: str
    embedding: Union[List[float], None, bool]
    meta: Optional[Dict[str, Any]]


class Retrieval(TypedDict):
    score: float
    doc: DocumentRecord


class AsyncDocumentAdder(Protocol):
    async def __call__(
        self,
        text: str,
        parent_id: Optional[DocumentId] = None,
        meta: Optional[Dict[str, Any]] = None,
        no_embedding: bool = False,
    ) -> DocumentId: ...


class AsyncDocumentDeleter(Protocol):
    async def __call__(self, doc_id: DocumentId) -> None: ...


class AsyncDocumentQuerier(abc.ABC):
    @abc.abstractmethod
    async def count(self) -> int: ...

    @abc.abstractmethod
    async def query_doc(
        self,
        doc_id: DocumentId,
        include_embedding: bool = False,
    ) -> DocumentRecord: ...

    @abc.abstractmethod
    async def query_children(
        self,
        doc_id: DocumentId,
        include_embedding: bool = False,
    ) -> List[DocumentRecord]: ...

    @abc.abstractmethod
    async def query_level(
        self,
        level: int,
        include_embedding: bool = False,
    ) -> List[DocumentRecord]: ...

    @abc.abstractmethod
    def dfs_traversal(
        self,
        include_embedding: bool = False,
    ) -> AsyncIterator[DocumentRecord]: ...


class DocumentAdder(Protocol):
    def __call__(
        self,
        text: str,
        parent_id: Optional[DocumentId] = None,
        meta: Optional[Dict[str, Any]] = None,
        no_embedding: bool = False,
    ) -> DocumentId: ...


class DocumentDeleter(Protocol):
    def __call__(self, doc_id: DocumentId) -> None: ...


class DocumentQuerier(abc.ABC):
    @abc.abstractmethod
    def count(self) -> int: ...

    @abc.abstractmethod
    def query_doc(
        self,
        doc_id: DocumentId,
        include_embedding: bool = False,
    ) -> DocumentRecord: ...

    @abc.abstractmethod
    def query_children(
        self,
        doc_id: DocumentId,
        include_embedding: bool = False,
    ) -> List[DocumentRecord]: ...

    @abc.abstractmethod
    def query_level(
        self,
        level: int,
        include_embedding: bool = False,
    ) -> List[DocumentRecord]: ...

    @abc.abstractmethod
    def dfs_traversal(
        self,
        include_embedding: bool = False,
    ) -> Iterator[DocumentRecord]: ...
