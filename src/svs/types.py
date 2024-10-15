from typing import (
    Any, Awaitable, Callable, Dict, List,
    Optional, Protocol, TypedDict, Union,
    AsyncIterator, Iterator, Tuple,
)

import networkx as nx  # type: ignore

import abc


EmbeddingFunc = Callable[[List[str]], Awaitable[List[List[float]]]]


DocumentId = int

EdgeId = int


NetworkXGraphTypes = Union[nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph]


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


class AsyncGraphInterface(abc.ABC):
    @abc.abstractmethod
    async def count_edges(self) -> int: ...

    @abc.abstractmethod
    async def add_directed_edge(
        self,
        from_doc: DocumentId,
        to_doc: DocumentId,
        relationship: DocumentId,
        weight: Optional[float] = None,
    ) -> EdgeId: ...

    @abc.abstractmethod
    async def add_edge(
        self,
        doc1: DocumentId,
        doc2: DocumentId,
        relationship: DocumentId,
        weight: Optional[float] = None,
    ) -> EdgeId: ...

    @abc.abstractmethod
    async def del_edge(self, edge_id: EdgeId) -> None: ...

    @abc.abstractmethod
    async def build_networkx_graph(
        self,
        multigraph: bool = True,
    ) -> NetworkXGraphTypes: ...


class AsyncKeyValueInterface(abc.ABC):
    @abc.abstractmethod
    async def has(self, key: str) -> bool: ...

    @abc.abstractmethod
    async def get(self, key: str, default: Any = KeyError) -> Any: ...

    @abc.abstractmethod
    async def set(self, key: str, val: Any) -> None: ...

    @abc.abstractmethod
    async def remove(self, key: str) -> None: ...

    @abc.abstractmethod
    async def count(self) -> int: ...

    @abc.abstractmethod
    def items(self) -> AsyncIterator[Tuple[str, Any]]: ...


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


class GraphInterface(abc.ABC):
    @abc.abstractmethod
    def count_edges(self) -> int: ...

    @abc.abstractmethod
    def add_directed_edge(
        self,
        from_doc: DocumentId,
        to_doc: DocumentId,
        relationship: DocumentId,
        weight: Optional[float] = None,
    ) -> EdgeId: ...

    @abc.abstractmethod
    def add_edge(
        self,
        doc1: DocumentId,
        doc2: DocumentId,
        relationship: DocumentId,
        weight: Optional[float] = None,
    ) -> EdgeId: ...

    @abc.abstractmethod
    def del_edge(self, edge_id: EdgeId) -> None: ...

    @abc.abstractmethod
    def build_networkx_graph(
        self,
        multigraph: bool = True,
    ) -> NetworkXGraphTypes: ...


class KeyValueInterface(abc.ABC):
    @abc.abstractmethod
    def has(self, key: str) -> bool: ...

    @abc.abstractmethod
    def __contains__(self, key: str) -> bool: ...

    @abc.abstractmethod
    def get(self, key: str, default: Any = KeyError) -> Any: ...

    @abc.abstractmethod
    def __getitem__(self, key: str) -> Any: ...

    @abc.abstractmethod
    def set(self, key: str, val: Any) -> None: ...

    @abc.abstractmethod
    def __setitem__(self, key: str, val: Any) -> None: ...

    @abc.abstractmethod
    def remove(self, key: str) -> None: ...

    @abc.abstractmethod
    def __delitem__(self, key: str) -> None: ...

    @abc.abstractmethod
    def count(self) -> int: ...

    @abc.abstractmethod
    def __len__(self) -> int: ...

    @abc.abstractmethod
    def items(self) -> Iterator[Tuple[str, Any]]: ...

    @abc.abstractmethod
    def __iter__(self) -> Iterator[str]: ...
