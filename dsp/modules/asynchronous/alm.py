from dsp.modules.lm import LM
from abc import ABC, abstractmethod


class ALM(LM, ABC):
    @abstractmethod
    async def __call__(
        self, prompt, only_completed=True, return_sorted=False, **kwargs
    ):
        pass

    @abstractmethod
    async def basic_request(self, prompt, **kwargs):
        pass
