from typing import List

from ..types import EmbeddingFunc


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
