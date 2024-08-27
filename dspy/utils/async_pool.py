from typing import List, TypeVar, Awaitable, AsyncGenerator
import asyncio
import traceback


class AsyncPool:
    T = TypeVar("T")

    class AsyncResult:
        def __init__(self):
            self.flag = asyncio.Event()
            self._val = None

        async def get(self):
            await self.flag.wait()
            return self._val

        def set(self, val):
            self._val = val
            self.flag.set()

    class AsyncResults:
        def __init__(self) -> None:
            self.results = []

        def __getitem__(self, key) -> "AsyncPool.AsyncResult":
            if key >= len(self.results):
                needed = key - len(self.results) + 1
                self.results.extend([AsyncPool.AsyncResult() for _ in range(needed)])
            return self.results[key]

    def __init__(self, max_workers):
        self.max_workers = max_workers
        self._queue = asyncio.Queue()
        self._tasks: List[asyncio.Task] = None
        self.run = False
        self.i = 0
        self.results = AsyncPool.AsyncResults()

    async def _worker(self):
        while self.run:
            task = await self._queue.get()
            i = self.i
            self.i += 1
            try:
                val = await task
            except Exception as e:
                traceback.print_exc()
                self.run = False
                raise e
            self._queue.task_done()
            self.results[i].set(val)

    async def __aenter__(self):
        self.run = True
        self._tasks = [
            asyncio.create_task(self._worker()) for _ in range(self.max_workers)
        ]
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self.run = False
        await self._queue.join()
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)

    def submit(self, task):
        self._queue.put_nowait(task)

    async def map(self, tasks: List[Awaitable[T]]) -> AsyncGenerator[T, None]:
        for task in tasks:
            self.submit(task)

        for i in range(len(tasks)):
            yield await self.results[i].get()
