import pytest
import asyncio
import random
import os

from svs.util import (
    locked,
    cached,
    file_cached_wget,
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
