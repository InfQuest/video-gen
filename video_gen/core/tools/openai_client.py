import asyncio
import functools
import json
import os
from typing import Any, Callable, Dict, List, Optional, Union

import httpx
from loguru import logger
from openai import AsyncOpenAI


def retry(func: Callable, retry_times: int = 5, base_delay_s: float = 2.0) -> Callable:
    """Retry decorator for async functions with exponential backoff."""

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        last_exc: Optional[BaseException] = None
        for i in range(retry_times):
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
            except Exception as e:
                last_exc = e
                logger.warning(f"{func.__name__} failed (attempt {i + 1}/{retry_times}): {e}")
                if i < retry_times - 1:
                    delay = min(base_delay_s * (2**i), 120.0)
                    await asyncio.sleep(delay)

        logger.error(f"{func.__name__} exhausted retries ({retry_times} attempts): {last_exc}")
        return None

    def wrapper_sync(*args, **kwargs):
        return asyncio.run(wrapper(*args, **kwargs))

    if asyncio.iscoroutinefunction(func):
        return wrapper
    else:
        return wrapper_sync


def default_openai_http_timeout() -> httpx.Timeout:
    """Long read timeout for large prompts / slow routing providers (e.g. OpenRouter + Gemini)."""
    read_s = float(os.environ.get("OPENAI_HTTP_READ_TIMEOUT", "900"))
    connect_s = float(os.environ.get("OPENAI_HTTP_CONNECT_TIMEOUT", "60"))
    return httpx.Timeout(
        connect=connect_s,
        read=read_s,
        write=min(read_s, 300.0),
        pool=connect_s,
    )


def extract_json_object(input_str: str, fixed_quotes: bool = False) -> Dict[str, Any]:
    """Extract a JSON object from a string with multiple fallback strategies.

    This function attempts to extract and parse JSON from various formats:
    - JSON in ```json code blocks
    - JSON in ``` code blocks (without json marker)
    - JSON between { and } (objects)
    - JSON between [ and ] (arrays)
    - Direct JSON string parsing

    Args:
        input_str (str): The input string to extract the JSON object from.
        fixed_quotes (bool, optional): Whether to fix unescaped quotes in the JSON object. Defaults to False.

    Returns:
        Dict[str, Any]: The extracted JSON object (dict or list).

    Raises:
        json.JSONDecodeError: If all extraction strategies fail.
    """
    if not input_str or not isinstance(input_str, str):
        raise json.JSONDecodeError("Input is empty or not a string", "", 0)

    # Strategy 1: Extract from ```json code block
    if "```json" in input_str:
        try:
            json_str = input_str.split("```json")[1].split("```")[0].strip()
            return json.loads(json_str)
        except (IndexError, json.JSONDecodeError):
            if fixed_quotes:
                try:
                    json_str = _fixed_unescaped_json_quotes(json_str)
                    return json.loads(json_str)
                except Exception:
                    pass  # Continue to next strategy

    # Strategy 2: Extract from ``` code block (without json marker)
    if "```" in input_str:
        try:
            parts = input_str.split("```")
            if len(parts) >= 3:
                json_str = parts[1].strip()
                return json.loads(json_str)
        except (IndexError, json.JSONDecodeError):
            pass  # Continue to next strategy

    # Strategy 3: Extract JSON object between { and }
    if "{" in input_str and "}" in input_str:
        try:
            start = input_str.find("{")
            end = input_str.rfind("}") + 1
            json_str = input_str[start:end]
            return json.loads(json_str)
        except json.JSONDecodeError:
            if fixed_quotes:
                try:
                    json_str = _fixed_unescaped_json_quotes(json_str)
                    return json.loads(json_str)
                except Exception:
                    pass  # Continue to next strategy

    # Strategy 4: Extract JSON array between [ and ]
    if "[" in input_str and "]" in input_str:
        try:
            start = input_str.find("[")
            end = input_str.rfind("]") + 1
            json_str = input_str[start:end]
            return json.loads(json_str)
        except json.JSONDecodeError:
            if fixed_quotes:
                try:
                    json_str = _fixed_unescaped_json_quotes(json_str)
                    return json.loads(json_str)
                except Exception:
                    pass  # Continue to next strategy

    # Strategy 5: Try parsing the whole string as JSON
    try:
        return json.loads(input_str.strip())
    except json.JSONDecodeError:
        if fixed_quotes:
            try:
                json_str = _fixed_unescaped_json_quotes(input_str.strip())
                return json.loads(json_str)
            except Exception:
                pass

    # All strategies failed
    raise json.JSONDecodeError("Failed to extract JSON from string", input_str, 0)


def _fixed_unescaped_json_quotes(input_str: str) -> str:
    """Fix unescaped quotes in a JSON string."""

    def start_with_any(x):
        return any(x.startswith(prefix) for prefix in (":", ",", "}", "]"))

    input_str = input_str.strip()
    result = ""
    in_quotes = False
    i = 0
    while i < len(input_str):
        c = input_str[i]
        if c == '"':
            if not in_quotes:
                in_quotes = True
                result += c
            else:
                if i == len(input_str) - 1 or start_with_any(input_str[i + 1 :].strip()):
                    result += c
                    in_quotes = False
                else:
                    result += "\\" + c
        else:
            if c == "\\":
                result += input_str[i : i + 2]
                i += 1
            else:
                result += c
        i += 1
    return result


class OpenAIClient:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://openrouter.ai/api/v1",
        model_name: str = "google/gemini-2.5-flash",
        **kwargs,
    ) -> None:
        self._model_name = model_name
        timeout: Union[float, httpx.Timeout, None] = kwargs.pop("timeout", None)
        if timeout is None:
            timeout = default_openai_http_timeout()
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout, **kwargs)

    @retry
    async def async_generate(self, instruction: str, user_input: str, **kwargs) -> str | None:
        """Generate a single-turn response from the model.

        Args:
            instruction (str): The system instruction to the model.
            user_input (str): The user input to the model.
            **kwargs: Additional arguments to pass to the OpenAI API.

        Returns:
            Optional[str]: The generated response from the model.
        """
        completion = await self._client.chat.completions.create(
            model=self._model_name,
            messages=[
                {"role": "system", "content": instruction},
                {"role": "user", "content": user_input},
            ],
            **kwargs,
        )

        if len(completion.choices) == 0:
            return None

        return completion.choices[0].message.content

    def generate(self, instruction: str, user_input: str, **kwargs) -> str | None:
        """The synchronous version of async_generate.

        Args:
            instruction (str): The system instruction to the model.
            user_input (str): The user input to the model.
            **kwargs: Additional arguments to pass to the OpenAI API.

        Returns:
            Optional[str]: The generated response from the model.
        """
        return asyncio.run(self.async_generate(instruction, user_input, **kwargs))

    def batch_generate(
        self,
        instruction: str | List[str],
        user_inputs: List[str],
        batch_size: int = 1000,
        **kwargs,
    ) -> List[Optional[str]]:
        """Generate responses from the model in batch.

        Args:
            instruction (str | List[str]): The system instruction to the model.
            user_inputs (List[str]): The user inputs to the model.
            batch_size (int): The number of user inputs to process in each batch. Defaults to 1000.
            **kwargs: Additional arguments to pass to the OpenAI API.

        Returns:
            List[Optional[str]]: The generated responses from the model.
        """

        if isinstance(instruction, str):
            instruction = [instruction] * len(user_inputs)

        assert len(instruction) == len(user_inputs), "The number of instructions and user inputs must be the same"

        async def batch_generate_coroutine():
            semaphore = asyncio.Semaphore(batch_size)

            async def generate_task(_instruction, user_input):
                async with semaphore:
                    return await self.async_generate(_instruction, user_input, **kwargs)

            tasks = [
                generate_task(_instruction, user_input) for _instruction, user_input in zip(instruction, user_inputs)
            ]
            results = await asyncio.gather(*tasks)

            return results

        return asyncio.run(batch_generate_coroutine())
