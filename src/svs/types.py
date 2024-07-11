from typing import (
    Callable, Awaitable, List, TypedDict, Optional,
    Dict, Any, Protocol,
)


EmbeddingFunc = Callable[[List[str]], Awaitable[List[List[float]]]]


DocumentId = int


class DocumentRecord(TypedDict):
    id: DocumentId
    parent_id: Optional[DocumentId]
    level: int
    text: str
    embedding: Optional[List[float]]
    meta: Optional[Dict[str, Any]]


class DocumentAdder(Protocol):
    async def __call__(
        self,
        text: str,
        parent_id: Optional[DocumentId] = None,
        meta: Optional[Dict[str, Any]] = None,
        no_embedding: bool = False,
    ) -> DocumentId: ...
