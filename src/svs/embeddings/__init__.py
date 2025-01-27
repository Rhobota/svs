from typing import Dict, Any

from ..types import EmbeddingFunc

from .util import (
    EMBEDDINGS_MAX_CACHE_SIZE,
    embedding_to_bytes,
    embedding_from_bytes,
    wrap_embeddings_func_check_magnitude,
)

from .mock import make_mock_embeddings_func

from .openai import make_openai_embeddings_func

from .ollama import make_ollama_embeddings_func


def make_embeddings_func(
    embedding_func_params: Dict[str, Any],
) -> EmbeddingFunc:
    embedding_func_params = { **embedding_func_params }  # shallow copy
    provider = embedding_func_params.pop('provider')
    if provider == 'mock':
        return make_mock_embeddings_func(**embedding_func_params)
    elif provider == 'openai':
        return make_openai_embeddings_func(**embedding_func_params)
    elif provider == 'ollama':
        return make_ollama_embeddings_func(**embedding_func_params)
    else:
        raise ValueError(f"unknown embedding provider name: {provider}")


__all__ = [
    'EMBEDDINGS_MAX_CACHE_SIZE',
    'embedding_to_bytes',
    'embedding_from_bytes',
    'wrap_embeddings_func_check_magnitude',
    'make_mock_embeddings_func',
    'make_openai_embeddings_func',
    'make_ollama_embeddings_func',
    'make_embeddings_func',
]
