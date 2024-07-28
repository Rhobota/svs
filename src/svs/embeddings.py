import aiohttp
import os
import struct
import functools

import numpy as np

from typing import List, Tuple, Optional, Dict, Any

from .util import cached

from .types import EmbeddingFunc


OPENAI_EMBEDDINGS_MAX_CACHE_SIZE = int(os.environ.get('OPENAI_EMBEDDINGS_MAX_CACHE_SIZE', 100))


def embedding_to_bytes(embedding: List[float]) -> bytes:
    return struct.pack(f'<{len(embedding)}f', *embedding)


def embedding_from_bytes(embedding: bytes) -> List[float]:
    size = struct.calcsize('<f')
    assert (len(embedding) % size) == 0
    n_items = len(embedding) // size
    return list(struct.unpack(f'<{n_items}f', embedding))


def make_embeddings_func(
    embedding_func_params: Dict[str, Any],
) -> EmbeddingFunc:
    embedding_func_params = { **embedding_func_params }  # shallow copy
    provider = embedding_func_params.pop('provider')
    if provider == 'mock':
        return make_mock_embeddings_func(**embedding_func_params)
    elif provider == 'openai':
        return make_openai_embeddings_func(**embedding_func_params)
    else:
        raise ValueError(f"unknown embedding provider name: {provider}")


def make_mock_embeddings_func() -> EmbeddingFunc:
    params = {
        'provider': 'mock',
    }

    async def mock_embeddings(
        list_of_strings: List[str],
    ) -> List[List[float]]:
        return [
            [1.0, 0.0, 0.0]
            for _ in list_of_strings
        ]

    setattr(mock_embeddings, '__embedding_func_params__', params)

    return mock_embeddings


def make_openai_embeddings_func(
    model: str = 'text-embedding-3-small',
    api_key: Optional[str] = None,
    dimensions: Optional[int] = None,
    user: Optional[str] = None,
) -> EmbeddingFunc:
    if api_key is None:
        api_key = os.environ['OPENAI_API_KEY']

    params = {
        'provider': 'openai',
        'model': model,
        'dimensions': dimensions,
    }

    async def openai_embeddings(
        list_of_strings: List[str],
    ) -> List[List[float]]:
        assert isinstance(list_of_strings, list)
        for s in list_of_strings:
            assert isinstance(s, str)

        results = await _cached_openai_embeddings_endpoint(
            api_key,
            tuple(list_of_strings),
            model,
            dimensions,
            user,
        )

        embeddings: List[List[float]] = []
        for i, d in enumerate(results['data']):
            embeddings.append(d['embedding'])
            assert i == d['index']
        assert len(embeddings) == len(list_of_strings)
        for e in embeddings:
            assert isinstance(e, list)
            for v in e:
                assert isinstance(v, float)
        return embeddings

    setattr(openai_embeddings, '__embedding_func_params__', params)

    return openai_embeddings


@cached(maxsize=OPENAI_EMBEDDINGS_MAX_CACHE_SIZE)
async def _cached_openai_embeddings_endpoint(
    api_key: Optional[str],
    tuple_of_strings: Tuple,
    model: str,
    dimensions: Optional[int],
    user: Optional[str],
) -> Any:
    url = 'https://api.openai.com/v1/embeddings'

    headers: Dict[str, Any] = {
        'Authorization': f'Bearer {api_key}',
    }
    payload: Dict[str, Any] = {
        'input': list(tuple_of_strings),
        'model': model,
        'encoding_format': 'float',
    }
    if dimensions is not None:
        payload['dimensions'] = dimensions
    if user is not None:
        payload['user'] = user

    async with aiohttp.ClientSession(raise_for_status=True) as session:
        async with session.post(url, headers=headers, json=payload) as response:
            return await response.json()


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
