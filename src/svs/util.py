import asyncio
import aiohttp

from collections import OrderedDict
import functools
import hashlib
import os
from urllib.parse import urlparse
from pathlib import Path

import numpy as np

from typing import (
    Optional, Union, Dict, List, Tuple,
    TypeVar, Callable, Awaitable,
)

from typing_extensions import ParamSpec

import logging

_LOG = logging.getLogger(__name__)


P = ParamSpec('P')  # <-- for generic programming
T = TypeVar('T')    # <-- for generic programming


def locked(
    lock: Union[asyncio.Lock, None] = None,
) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Awaitable[T]]]:
    """
    Decorator for async functions to lock them so that they can only be
    executed *serially* (rather than *currently*).
    """
    if lock is None:
        lock = asyncio.Lock()
    def decorator(wrapped: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        @functools.wraps(wrapped)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            async with lock:
                return await wrapped(*args, **kwargs)
        return wrapper
    return decorator


def cached(
    maxsize: Optional[int] = None,
) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Awaitable[T]]]:
    """
    Cache an async function with an LRU cache.

    Pass `maxsize` as a positive integer to set the maximum cache size.
    Pass `maxsize = None` to make the cache grow indefinitely.

    This decorator also ensure that calls to the wrapped async function
    are not run concurrently for the *same* input.
    """
    def decorator(wrapped: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        cache: OrderedDict[int, T] = OrderedDict()
        events: Dict[int, asyncio.Event] = {}
        @functools.wraps(wrapped)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            key = hash((args, tuple(sorted(kwargs.items()))))
            while True:
                if key in cache:
                    _LOG.info(f"cached({key}): CACHE HIT")
                    cache.move_to_end(key)
                    return cache[key]
                if key in events:
                    event = events[key]
                    _LOG.info(f"cached({key}): avoiding concurrency")
                    await event.wait()
                else:
                    event = asyncio.Event()
                    events[key] = event
                    _LOG.info(f"cached({key}): cache miss ... will compute")
                    try:
                        res = await wrapped(*args, **kwargs)
                        cache[key] = res
                        if maxsize is not None and len(cache) > maxsize:
                            cache.popitem(last=False)
                    finally:
                        event.set()
                        del events[key]
                    return res
        return wrapper
    return decorator


@locked()
async def file_cached_wget(url: str) -> Path:
    """
    HTTP _get_ the resource at `url`, and cache the results in the local
    file-system for subsequent calls to get the same `url` (all in an
    asyncio-friendly way). Returns the path to the locally-stored file.

    This function is locked so you can only get one `url` at a time. That
    is a heavy-handed way to deal with the race-condition so we don't get
    the *same* `url` concurrently, with the obvious less-than-ideal outcome
    that we cannot get two *different* `url`s concurrently either (although
    for the use-case we care about that's fine). Maybe that will be loosed
    in the future...
    """
    loop = asyncio.get_running_loop()
    hash = hashlib.md5(url.encode()).hexdigest()
    extension = os.path.splitext(urlparse(url).path)[1]
    path = Path('.remote_cache') / Path(f'{hash}{extension}')
    def _check_exists() -> bool:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return os.path.exists(path)
    exists = await loop.run_in_executor(None, _check_exists)
    if exists:
        _LOG.info(f"file_cached_wget({repr(url)}): CACHE HIT")
        return path
    _LOG.info(f"file_cached_wget({repr(url)}): cache miss ... will *get*")
    f = await loop.run_in_executor(None, open, path, 'wb')
    closed: bool = False
    try:
        async with aiohttp.ClientSession(raise_for_status=True) as session:
            async with session.get(url) as response:
                async for data in response.content.iter_chunked(4096 * 4096):
                    await loop.run_in_executor(None, f.write, data)
        await loop.run_in_executor(None, f.close)
        closed = True
        return path
    except Exception as e:
        try:
            # It's important we don't accidentally leave opened and/or
            # partially-written files!
            if not closed:
                await loop.run_in_executor(None, f.close)
                closed = True
            await loop.run_in_executor(None, os.unlink, path)
        except Exception as e2:
            # We tried our best; nothing we can do now except log this.
            _LOG.exception(e2)
        raise e


def get_top_k(scores: np.ndarray, top_k: int) -> List[Tuple[float, int]]:
    """
    As efficiently as possible, find the `top_k` scores in the `scores` array.
    Returns a list of length `top_k` of tuples:
      (score, index)
    """
    assert scores.ndim == 1
    assert isinstance(top_k, int)
    if top_k > len(scores):
        top_k = len(scores)
    if top_k <= 0:
        return []
    indices = np.argpartition(scores, -top_k)[-top_k:]
    return sorted([(float(scores[i]), int(i)) for i in indices], reverse=True)


def chunkify(seq: List[T], n: int) -> List[List[T]]:
    """Split `seq` into sublists of size `n`"""
    if n <= 0:
        raise ValueError('n must be positive')
    return [seq[i * n:(i + 1) * n] for i in range((len(seq) + n - 1) // n)]
