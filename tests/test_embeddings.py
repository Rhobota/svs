import pytest

from svs.embeddings import (
    embedding_to_bytes,
    embedding_from_bytes,
    make_embeddings_func,
    wrap_embeddings_func_check_magnitude,
)

from typing import List


def test_embedding_to_bytes():
    assert embedding_to_bytes([]) == b''
    assert embedding_to_bytes([1.0]) == b'\x00\x00\x80?'
    assert embedding_to_bytes([1.0, 3.5]) == b'\x00\x00\x80?\x00\x00`@'


def test_embedding_from_bytes():
    assert embedding_from_bytes(b'') == []
    assert embedding_from_bytes(b'\x00\x00\x80?') == [1.0]
    assert embedding_from_bytes(b'\x00\x00\x80?\x00\x00`@') == [1.0, 3.5]


@pytest.mark.asyncio
async def test_make_embeddings_func():
    mock = make_embeddings_func({
        'provider': 'mock',
    })
    assert getattr(mock, '__embedding_func_params__') == {
        'provider': 'mock',
    }
    assert (await mock(['anything'])) == [[1.0, 0.0, 0.0]]

    openai = make_embeddings_func({
        'provider': 'openai',
        'model': 'text-embedding-3-small',
        'api_key': '...',
    })
    assert getattr(openai, '__embedding_func_params__') == {
        'provider': 'openai',
        'model': 'text-embedding-3-small',
        'dimensions': None,
    }

    with pytest.raises(ValueError):
        make_embeddings_func({
            'provider': 'INVALID',
        })


@pytest.mark.asyncio
async def test_wrap_embeddings_func_check_magnitude():
    async def embedding_func_1(list_of_texts: List[str]) -> List[List[float]]:
        return [
            [1.0, 0.1, 0.0]  # <-- magnitude too large
            for _ in list_of_texts
        ]
    async def embedding_func_2(list_of_texts: List[str]) -> List[List[float]]:
        return [
            [0.99, 0.0, 0.0]  # <-- magnitude too small
            for _ in list_of_texts
        ]
    async def embedding_func_3(list_of_texts: List[str]) -> List[List[float]]:
        return [
            [1.0, 0.0, 0.0]  # <-- magnitude just right!
            for _ in list_of_texts
        ]
    tolerance = 0.001
    wrapped_1 = wrap_embeddings_func_check_magnitude(embedding_func_1, tolerance)
    wrapped_2 = wrap_embeddings_func_check_magnitude(embedding_func_2, tolerance)
    wrapped_3 = wrap_embeddings_func_check_magnitude(embedding_func_3, tolerance)
    with pytest.raises(ValueError):
        await wrapped_1(['anything'])
    with pytest.raises(ValueError):
        await wrapped_2(['anything'])
    assert (await wrapped_3(['anything'])) == [[1.0, 0.0, 0.0]]
