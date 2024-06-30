import aiohttp

from typing import List, Optional, Dict, Any

from .types import EmbeddingFunc


def make_openai_embeddings_func(
    model: str = 'text-embedding-3-small',
    dimensions: Optional[int] = None,
    user: Optional[str] = None,
) -> EmbeddingFunc:
    async def openai_embeddings(
        list_of_strings: List[str],
    ) -> List[List[float]]:
        assert isinstance(list_of_strings, list)
        for s in list_of_strings:
            assert isinstance(s, str)

        url = 'https://api.openai.com/v1/embeddings'

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
            async with session.post(url, json=payload) as response:
                results = await response.json()

        embeddings: List[List[float]] = []
        for i, d in enumerate(results):
            embeddings.append(d['embedding'])
            assert i == d['index']
        assert len(embeddings) == len(list_of_strings)
        for e in embeddings:
            assert isinstance(e, list)
            for v in e:
                assert isinstance(v, float)
        return embeddings

    return openai_embeddings
