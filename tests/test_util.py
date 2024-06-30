import pytest
import asyncio
import random
import os

import numpy as np

from svs.util import (
    locked,
    cached,
    file_cached_wget,
    get_top_k,
)


@pytest.mark.asyncio
async def test_locked():
    is_inside = False
    @locked()
    async def _something(val: int) -> int:
        nonlocal is_inside
        assert is_inside is False
        is_inside = True
        await asyncio.sleep(0.1)
        is_inside = False
        return val
    await asyncio.gather(
        _something(2),
        _something(3),
    )


@pytest.mark.asyncio
async def test_cached():
    @cached()
    async def _something(val: int) -> int:
        await asyncio.sleep(0.1)
        return val + random.randint(0, 10000000)
    res = await asyncio.gather(
        _something(2),
        _something(3),
        _something(2),
        _something(3),
    )
    assert res[0] == res[2]
    assert res[1] == res[3]


@pytest.mark.asyncio
async def test_file_cached_wget():
    url = 'https://raw.githubusercontent.com/Rhobota/svs/main/logos/svs.png'
    cache_location = '.remote_cache/2cff95e6fe5a3de0e7ff3270dae85f6c'

    if os.path.exists(cache_location):
        os.unlink(cache_location)
        assert not os.path.exists(cache_location)

    data1 = await file_cached_wget(url)
    assert len(data1) == 23123
    assert os.path.exists(cache_location)

    data2 = await file_cached_wget(url)
    assert data1 == data2


def test_get_top_k():
    scores = np.array([])

    top = list(get_top_k(scores, top_k=0))
    assert len(top) == 0

    top = list(get_top_k(scores, top_k=1))
    assert len(top) == 0

    scores = np.array([0.4])

    top = list(get_top_k(scores, top_k=0))
    assert len(top) == 0

    top = list(get_top_k(scores, top_k=1))
    assert len(top) == 1
    s, i = top[0]
    assert s == 0.4
    assert i == 0

    top = list(get_top_k(scores, top_k=2))
    assert len(top) == 1
    s, i = top[0]
    assert s == 0.4
    assert i == 0

    scores = np.array([0.4, 0.2])

    top = list(get_top_k(scores, top_k=0))
    assert len(top) == 0

    top = list(get_top_k(scores, top_k=1))
    assert len(top) == 1
    s, i = top[0]
    assert s == 0.4
    assert i == 0

    top = list(get_top_k(scores, top_k=2))
    assert len(top) == 2
    s, i = top[0]
    assert s == 0.4
    assert i == 0
    s, i = top[1]
    assert s == 0.2
    assert i == 1

    top = list(get_top_k(scores, top_k=3))
    assert len(top) == 2
    s, i = top[0]
    assert s == 0.4
    assert i == 0
    s, i = top[1]
    assert s == 0.2
    assert i == 1

    scores = np.array([0.2, 0.4])

    top = list(get_top_k(scores, top_k=0))
    assert len(top) == 0

    top = list(get_top_k(scores, top_k=1))
    assert len(top) == 1
    s, i = top[0]
    assert s == 0.4
    assert i == 1

    top = list(get_top_k(scores, top_k=2))
    assert len(top) == 2
    s, i = top[0]
    assert s == 0.4
    assert i == 1
    s, i = top[1]
    assert s == 0.2
    assert i == 0

    top = list(get_top_k(scores, top_k=3))
    assert len(top) == 2
    s, i = top[0]
    assert s == 0.4
    assert i == 1
    s, i = top[1]
    assert s == 0.2
    assert i == 0

    scores = np.array([0.4, 0.2, 0.8])

    top = list(get_top_k(scores, top_k=0))
    assert len(top) == 0

    top = list(get_top_k(scores, top_k=1))
    assert len(top) == 1
    s, i = top[0]
    assert s == 0.8
    assert i == 2

    top = list(get_top_k(scores, top_k=2))
    assert len(top) == 2
    s, i = top[0]
    assert s == 0.8
    assert i == 2
    s, i = top[1]
    assert s == 0.4
    assert i == 0

    top = list(get_top_k(scores, top_k=3))
    assert len(top) == 3
    s, i = top[0]
    assert s == 0.8
    assert i == 2
    s, i = top[1]
    assert s == 0.4
    assert i == 0
    s, i = top[2]
    assert s == 0.2
    assert i == 1

    top = list(get_top_k(scores, top_k=4))
    assert len(top) == 3
    s, i = top[0]
    assert s == 0.8
    assert i == 2
    s, i = top[1]
    assert s == 0.4
    assert i == 0
    s, i = top[2]
    assert s == 0.2
    assert i == 1

    scores = np.array([0.4, 0.8, 0.2])

    top = list(get_top_k(scores, top_k=0))
    assert len(top) == 0

    top = list(get_top_k(scores, top_k=1))
    assert len(top) == 1
    s, i = top[0]
    assert s == 0.8
    assert i == 1

    top = list(get_top_k(scores, top_k=2))
    assert len(top) == 2
    s, i = top[0]
    assert s == 0.8
    assert i == 1
    s, i = top[1]
    assert s == 0.4
    assert i == 0

    top = list(get_top_k(scores, top_k=3))
    assert len(top) == 3
    s, i = top[0]
    assert s == 0.8
    assert i == 1
    s, i = top[1]
    assert s == 0.4
    assert i == 0
    s, i = top[2]
    assert s == 0.2
    assert i == 2

    top = list(get_top_k(scores, top_k=4))
    assert len(top) == 3
    s, i = top[0]
    assert s == 0.8
    assert i == 1
    s, i = top[1]
    assert s == 0.4
    assert i == 0
    s, i = top[2]
    assert s == 0.2
    assert i == 2

    scores = np.array([0.8, 0.4, 0.2])

    top = list(get_top_k(scores, top_k=0))
    assert len(top) == 0

    top = list(get_top_k(scores, top_k=1))
    assert len(top) == 1
    s, i = top[0]
    assert s == 0.8
    assert i == 0

    top = list(get_top_k(scores, top_k=2))
    assert len(top) == 2
    s, i = top[0]
    assert s == 0.8
    assert i == 0
    s, i = top[1]
    assert s == 0.4
    assert i == 1

    top = list(get_top_k(scores, top_k=3))
    assert len(top) == 3
    s, i = top[0]
    assert s == 0.8
    assert i == 0
    s, i = top[1]
    assert s == 0.4
    assert i == 1
    s, i = top[2]
    assert s == 0.2
    assert i == 2

    top = list(get_top_k(scores, top_k=4))
    assert len(top) == 3
    s, i = top[0]
    assert s == 0.8
    assert i == 0
    s, i = top[1]
    assert s == 0.4
    assert i == 1
    s, i = top[2]
    assert s == 0.2
    assert i == 2

    scores = np.array([0.8, 0.2, 0.4])

    top = list(get_top_k(scores, top_k=0))
    assert len(top) == 0

    top = list(get_top_k(scores, top_k=1))
    assert len(top) == 1
    s, i = top[0]
    assert s == 0.8
    assert i == 0

    top = list(get_top_k(scores, top_k=2))
    assert len(top) == 2
    s, i = top[0]
    assert s == 0.8
    assert i == 0
    s, i = top[1]
    assert s == 0.4
    assert i == 2

    top = list(get_top_k(scores, top_k=3))
    assert len(top) == 3
    s, i = top[0]
    assert s == 0.8
    assert i == 0
    s, i = top[1]
    assert s == 0.4
    assert i == 2
    s, i = top[2]
    assert s == 0.2
    assert i == 1

    top = list(get_top_k(scores, top_k=4))
    assert len(top) == 3
    s, i = top[0]
    assert s == 0.8
    assert i == 0
    s, i = top[1]
    assert s == 0.4
    assert i == 2
    s, i = top[2]
    assert s == 0.2
    assert i == 1
