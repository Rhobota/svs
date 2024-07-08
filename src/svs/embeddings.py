import aiohttp
import os
import struct

from typing import List, Optional, Dict, Any

from .types import EmbeddingFunc


def embedding_to_bytes(embedding: List[float]) -> bytes:
    return struct.pack(f'<{len(embedding)}f', *embedding)


def embedding_from_bytes(embedding: bytes) -> List[float]:
    return list(struct.unpack_from('<f', embedding))


def make_embeddings_func(
    factory_name: str,
    factory_params: Dict[str, Any],
) -> EmbeddingFunc:
    if factory_name == 'openai':
        return make_openai_embeddings_func(**factory_params)
    else:
        raise ValueError(f"unknown facotry name: {factory_name}")


def make_openai_embeddings_func(
    model: str = 'text-embedding-3-small',
    api_key: Optional[str] = None,
    dimensions: Optional[int] = None,
    user: Optional[str] = None,
) -> EmbeddingFunc:
    if api_key is None:
        api_key = os.environ['OPENAI_API_KEY']
    async def openai_embeddings(
        list_of_strings: List[str],
    ) -> List[List[float]]:
        assert isinstance(list_of_strings, list)
        for s in list_of_strings:
            assert isinstance(s, str)

        url = 'https://api.openai.com/v1/embeddings'

        headers: Dict[str, Any] = {
            'Authorization': f'Bearer {api_key}',
        }
        payload: Dict[str, Any] = {
            'input': list_of_strings,
            'model': model,
            'encoding_format': 'float',
        }
        if dimensions is not None:
            payload['dimensions'] = dimensions
        if user is not None:
            payload['user'] = user

        async with aiohttp.ClientSession(raise_for_status=True) as session:
            async with session.post(url, headers=headers, json=payload) as response:
                results = await response.json()

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

    return openai_embeddings
