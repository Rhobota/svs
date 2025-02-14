import aiohttp
import os

from typing import List, Tuple, Optional, Dict, Any

from .util import EMBEDDINGS_MAX_CACHE_SIZE

from ..util import cached

from ..types import EmbeddingFunc


def make_openai_embeddings_func(
    model: str = 'text-embedding-3-small',
    api_key: Optional[str] = None,
    dimensions: Optional[int] = None,
    user: Optional[str] = None,
) -> EmbeddingFunc:
    if api_key is None:
        api_key = os.environ.get('OPENAI_API_KEY', None)

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

        if api_key is None:
            raise RuntimeError('No OpenAI API key found! It was not passed to the function nor was it in the OPENAI_API_KEY environment variable.')

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


@cached(maxsize=EMBEDDINGS_MAX_CACHE_SIZE)
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

    async with aiohttp.ClientSession(raise_for_status=False) as session:
        async with session.post(url, headers=headers, json=payload) as response:
            status = response.status
            data = await response.json()
            if status != 200:
                message = data.get('error', {}).get('message', str(data))
                raise RuntimeError(f'OpenAI API error: status={status}, message={message}')
            return data
