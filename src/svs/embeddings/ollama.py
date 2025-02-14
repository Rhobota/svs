import aiohttp
import json
import os

from typing import List, Tuple, Dict, Any, Union

from .util import EMBEDDINGS_MAX_CACHE_SIZE

from ..util import cached

from ..types import EmbeddingFunc


def make_ollama_embeddings_func(
    model: str,
    truncate: bool = True,
    keep_alive: str = '5m',
    base_url: Union[str, None] = None,
) -> EmbeddingFunc:
    params = {
        'provider': 'ollama',
        'model': model,
        'truncate': truncate,
        'keep_alive': keep_alive,
        'base_url': base_url,
    }

    async def ollama_embeddings(
        list_of_strings: List[str],
    ) -> List[List[float]]:
        assert isinstance(list_of_strings, list)
        for s in list_of_strings:
            assert isinstance(s, str)

        base_url_to_use = base_url if base_url else os.environ.get('OLLAMA_BASE_URL', 'http://127.0.0.1:11434')

        data = await _cached_ollama_embeddings_endpoint(
            base_url_to_use,
            tuple(list_of_strings),
            model,
            truncate,
            keep_alive,
        )

        embeddings: List[List[float]] = data['embeddings']

        assert isinstance(embeddings, list)
        assert len(embeddings) == len(list_of_strings)

        for e in embeddings:
            assert isinstance(e, list)
            for v in e:
                assert isinstance(v, float)

        return embeddings

    setattr(ollama_embeddings, '__embedding_func_params__', params)

    return ollama_embeddings


@cached(maxsize=EMBEDDINGS_MAX_CACHE_SIZE)
async def _cached_ollama_embeddings_endpoint(
    base_url: str,
    tuple_of_strings: Tuple,
    model: str,
    truncate: bool,
    keep_alive: str,
) -> Any:
    url = f'{base_url}/api/embed'

    payload: Dict[str, Any] = {
        'model': model,
        'truncate': truncate,
        'keep_alive': keep_alive,
        # 'options': {},  <-- maybe support this in the future?
        'input': list(tuple_of_strings),
    }

    async with aiohttp.ClientSession(raise_for_status=False) as session:
        async with session.post(url, json=payload) as response:
            status = response.status
            data = await response.json()

            if status != 200:
                error_text: str
                try:
                    error_text = data['error']
                except:
                    error_text = f'status={status}: {json.dumps(data)}'
                raise RuntimeError(f'Ollama error: {error_text}')

            return data
