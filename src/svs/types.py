from typing import Callable, Awaitable, List


EmbeddingFunc = Callable[[List[str]], Awaitable[List[List[float]]]]
