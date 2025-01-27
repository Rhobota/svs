import os
import struct
import functools

import numpy as np

from typing import List

from ..types import EmbeddingFunc


EMBEDDINGS_MAX_CACHE_SIZE = int(os.environ.get('EMBEDDINGS_MAX_CACHE_SIZE', 100))


def embedding_to_bytes(embedding: List[float]) -> bytes:
    return struct.pack(f'<{len(embedding)}f', *embedding)


def embedding_from_bytes(embedding: bytes) -> List[float]:
    size = struct.calcsize('<f')
    assert (len(embedding) % size) == 0
    n_items = len(embedding) // size
    return list(struct.unpack(f'<{n_items}f', embedding))


def wrap_embeddings_func_check_magnitude(
    embedding_func: EmbeddingFunc,
    tolerance: float,
) -> EmbeddingFunc:
    @functools.wraps(embedding_func)
    async def wrapped(
        list_of_strings: List[str],
    ) -> List[List[float]]:
        vectors = await embedding_func(list_of_strings)
        vectors_np = np.array(vectors, dtype=np.float32)
        mags = np.sqrt((vectors_np * vectors_np).sum(axis=1))
        if (np.abs(mags - 1.0) > tolerance).any():
            raise ValueError("embedding magnitude out of spec")
        return vectors

    return wrapped
