from typing import List, TypeVar, Awaitable, AsyncGenerator
import asyncio


class AsyncPool:
    T = TypeVar("T")

    class AsyncResult:
        """
        Helper class to manage waiting for results and setting them.
        """

        def __init__(self, pool: "AsyncPool", idx: int) -> None:
            self.flag = asyncio.Event()
            self._val = None
            self._pool = pool
            self.idx = idx

        async def get(self):
            await self.flag.wait()
            return self._val

        def set(self, val):
            self._val = val
            self.flag.set()
            self._pool._result_queue.put_nowait(self)

    class AsyncResults:
        """
        Helper class to manage creation and consumption of results.
        """

        def __init__(self, pool: "AsyncPool") -> None:
            self.results = []
            self.consumed = 0
            self.pool = pool

        def __getitem__(self, key) -> "AsyncPool.AsyncResult":
            if key >= len(self.results):
                needed = key - len(self.results) + 1
                new_futures = [
                    AsyncPool.AsyncResult(self.pool, len(self.results) + i)
                    for i in range(needed)
                ]
                self.results.extend(new_futures)
            return self.results[key]

    def __init__(self, max_workers):
        self.max_workers = max_workers
        self._tasks: List[asyncio.Task] = None
        self._queue = asyncio.Queue()  # task queue
        self._result_queue = asyncio.Queue()  # result queue for unordered results
        self.results = AsyncPool.AsyncResults(self)  # result buffer for ordered results

        # stats
        self.run = False
        self.i = 0
        self.total = 0

        # exceptions
        self.exception_flag = asyncio.Event()
        self.stored_exception = None

    async def _worker(self):
        """
        Driving worker loop
        """
        while self.run:
            task = await self._queue.get()
            i = self.i
            self.i += 1
            try:
                val = await task
            except Exception as e:
                self.run = False
                self.exception_flag.set()
                self.stored_exception = e
                break
            self._queue.task_done()
            self.results[i].set(val)

    async def __aenter__(self):
        self.run = True
        self._tasks = [
            asyncio.create_task(self._worker()) for _ in range(self.max_workers)
        ]
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if not self.exception_flag.is_set():
            # exception flag not set, but maybe not all tasks are completed
            # as such, we need to wait for all tasks to complete
            join_task = asyncio.create_task(self._queue.join())
            exception_task = asyncio.create_task(self.exception_flag.wait())

            await asyncio.wait(
                [join_task, exception_task],
                return_when=asyncio.FIRST_COMPLETED,
            )

            if self.exception_flag.is_set():
                join_task.cancel()
                raise self.stored_exception

        # stop all workers now that we are done w/ all tasks
        self.run = False
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)

    def submit(self, task: Awaitable[T]) -> None:
        """
        Submit a coroutine to be executed by the pool.

        Args:
            task (Awaitable[T]): coroutine to execute
        """
        self._queue.put_nowait(task)
        self.total += 1

    async def _map(
        self, tasks: List[Awaitable[T]], unordered: bool
    ) -> AsyncGenerator[T, None]:
        """
        Returns a generator that yields the results of the tasks in the same order
        they were submitted or in the order they complete if unordered
        """
        for task in tasks:
            self.submit(task)

        for i in range(len(tasks)):
            if not unordered:
                get_task = asyncio.create_task(self.results[i].get())
            else:
                get_task = asyncio.create_task(self._result_queue.get())
            exception_task = asyncio.create_task(self.exception_flag.wait())
            await asyncio.wait(
                [get_task, exception_task],
                return_when=asyncio.FIRST_COMPLETED,
            )

            # it was an exception
            if self.exception_flag.is_set():
                get_task.cancel()
                break
            # it was a success (done with the result)
            exception_task.cancel()
            if not unordered:
                yield get_task.result()
            else:
                yield get_task.result()._val

        # if there was an exception, re-raise it in main thread
        if self.stored_exception is not None:
            raise self.stored_exception

    async def map(self, tasks: List[Awaitable[T]]) -> AsyncGenerator[T, None]:
        """
        Returns a generator that yields the results of the tasks in the same order
        they were submitted

        Args:
            tasks (List[Awaitable[T]]): list of coroutines to run.

        Yields:
            T: the results of the tasks, in order
        """
        async for elem in self._map(tasks, unordered=False):
            yield elem

    async def as_completed(self, tasks: List[Awaitable[T]]) -> AsyncGenerator[T, None]:
        """
        Returns a generator that yields the results of the tasks in the order in which
        they first complete

        Args:
            tasks (List[Awaitable[T]]): list of coroutines to run.

        Yields:
            T: the results of the tasks, in arbitrary order
        """
        async for elem in self._map(tasks, unordered=True):
            yield elem
