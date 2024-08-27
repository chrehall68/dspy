from typing import Any, Optional, Literal
import httpx
from dsp.modules.asynchronous.alm import ALM
import openai
from dsp.modules.cache_utils import cache_turn_on
import json
import backoff
from dsp.utils.settings import settings
import logging


def backoff_hdlr(details):
    """Handler from https://pypi.org/project/backoff/"""
    print(
        "Backing off {wait:0.1f} seconds after {tries} tries "
        "calling function {target} with kwargs "
        "{kwargs}".format(**details),
    )


_async_client: openai.AsyncOpenAI = None
ERRORS = (openai.RateLimitError,)


class AGPT3(ALM):

    def __init__(
        self,
        model: str = "gpt-3.5-turbo-instruct",
        api_key: Optional[str] = None,
        api_provider: Literal["openai"] = "openai",
        api_base: Optional[str] = None,
        base_url: Optional[str] = None,
        model_type: Literal["chat", "text"] = None,
        system_prompt: Optional[str] = None,
        http_client: Optional[httpx.Client] = None,
        default_headers: Optional[dict[str, str]] = None,
        **kwargs,
    ):
        global _async_client
        super().__init__(model)
        self.provider = "openai"
        _async_client = openai.AsyncOpenAI()
        _async_client.api_type = api_provider

        self.system_prompt = system_prompt

        assert (
            api_provider != "azure"
        ), "Azure functionality with base OpenAI has been deprecated, please use dspy.AzureOpenAI instead."

        default_model_type = (
            "chat"
            if ("gpt-3.5" in model or "turbo" in model or "gpt-4" in model)
            and ("instruct" not in model)
            else "text"
        )
        self.model_type = model_type if model_type else default_model_type

        if api_key:
            _async_client.api_key = api_key
        api_base = base_url or api_base
        if api_base:
            _async_client.base_url = api_base
        if http_client:
            _async_client.http_client = http_client

        self.kwargs = {
            "temperature": 0.0,
            "max_tokens": 150,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "n": 1,
            **kwargs,
        }  # TODO: add kwargs above for </s>

        self.kwargs["model"] = model
        self.history: list[dict[str, Any]] = []

    async def __call__(
        self,
        prompt: str,
        only_completed: bool = True,
        return_sorted: bool = False,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """Retrieves completions from GPT-3.

        Args:
            prompt (str): prompt to send to GPT-3
            only_completed (bool, optional): return only completed responses and ignores completion due to length. Defaults to True.
            return_sorted (bool, optional): sort the completion choices using the returned probabilities. Defaults to False.

        Returns:
            list[dict[str, Any]]: list of completion choices
        """

        assert only_completed, "for now"
        assert return_sorted is False, "for now"

        # if kwargs.get("n", 1) > 1:
        #     if self.model_type == "chat":
        #         kwargs = {**kwargs}
        #     else:
        #         kwargs = {**kwargs, "logprobs": 5}

        response = await self.request(prompt, **kwargs)

        self.log_usage(response)
        choices = response["choices"]

        completed_choices = [c for c in choices if c["finish_reason"] != "length"]

        if only_completed and len(completed_choices):
            choices = completed_choices

        if kwargs.get("logprobs", False):
            completions = [
                {"text": self._get_choice_text(c), "logprobs": c["logprobs"]}
                for c in choices
            ]
        else:
            completions = [self._get_choice_text(c) for c in choices]

        if return_sorted and kwargs.get("n", 1) > 1:
            scored_completions = []

            for c in choices:
                tokens, logprobs = (
                    c["logprobs"]["tokens"],
                    c["logprobs"]["token_logprobs"],
                )

                if "<|endoftext|>" in tokens:
                    index = tokens.index("<|endoftext|>") + 1
                    tokens, logprobs = tokens[:index], logprobs[:index]

                avglog = sum(logprobs) / len(logprobs)
                scored_completions.append((avglog, self._get_choice_text(c), logprobs))
            scored_completions = sorted(scored_completions, reverse=True)
            if logprobs:
                completions = [
                    {"text": c, "logprobs": lp} for _, c, lp in scored_completions
                ]
            else:
                completions = [c for _, c in scored_completions]

        return completions

    async def basic_request(self, prompt: str, **kwargs):
        raw_kwargs = kwargs

        kwargs = {**self.kwargs, **kwargs}
        if self.model_type == "chat":
            # caching mechanism requires hashable kwargs
            messages = [{"role": "user", "content": prompt}]
            if self.system_prompt:
                messages.insert(0, {"role": "system", "content": self.system_prompt})
            kwargs["messages"] = messages
            kwargs = {"stringify_request": json.dumps(kwargs)}
            response = await chat_request(**kwargs)

        else:
            kwargs["prompt"] = prompt
            kwargs = {"stringify_request": json.dumps(kwargs)}
            response = completions_request(**kwargs)

        history = {
            "prompt": prompt,
            "response": response,
            "kwargs": kwargs,
            "raw_kwargs": raw_kwargs,
        }
        self.history.append(history)

        return response

    @backoff.on_exception(
        backoff.expo,
        ERRORS,
        max_time=settings.backoff_time,
        on_backoff=backoff_hdlr,
    )
    async def request(self, prompt, **kwargs):
        return await self.basic_request(prompt, **kwargs)

    def log_usage(self, response):
        """Log the total tokens from the OpenAI API response."""
        usage_data = response.get("usage")
        if usage_data:
            total_tokens = usage_data.get("total_tokens")
            logging.debug(f"OpenAI Response Token Usage: {total_tokens}")

    def _get_choice_text(self, choice: dict[str, Any]) -> str:
        if self.model_type == "chat":
            return choice["message"]["content"]
        return choice["text"]


async def chat_request(stringify_request):
    global _async_client
    if not hasattr(chat_request, "_cache"):
        chat_request._cache = {}
    if stringify_request in chat_request._cache and cache_turn_on:
        return chat_request._cache[stringify_request]

    kwargs = json.loads(stringify_request)
    result = await _async_client.chat.completions.create(**kwargs)

    if cache_turn_on:
        chat_request._cache[stringify_request] = result.model_dump()
    return result.model_dump()


async def completions_request(stringify_request):
    global _async_client
    if not hasattr(completions_request, "_cache"):
        completions_request._cache = {}
    if stringify_request in completions_request._cache and cache_turn_on:
        return completions_request._cache[stringify_request]

    kwargs = json.loads(stringify_request)
    result = await _async_client.completions.create(**kwargs)

    if cache_turn_on:
        completions_request._cache[stringify_request] = result.model_dump()
    return result.model_dump()
