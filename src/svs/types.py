from typing import (
    Callable, Awaitable, List, TypedDict, Optional,
    Dict, Any,
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
