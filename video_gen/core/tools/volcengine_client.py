"""Volcengine ARK API client for image generation.

This module provides a client for the Volcengine ARK API, specifically for
image generation using models like doubao-seedream-4-0-250828.
"""

import asyncio
from typing import Literal

import requests
from loguru import logger
from pydantic import BaseModel

from video_gen.core.configs.config import settings


class ImageGenerationRequest(BaseModel):
    """Request model for image generation."""

    model: str = "doubao-seedream-4-0-250828"
    prompt: str
    size: str = "2K"  # Can be "1K"/"2K"/"4K" or "WIDTHxHEIGHT" (e.g., "2560x1440")
    sequential_image_generation: Literal["enabled", "disabled"] = "disabled"
    stream: bool = False
    response_format: Literal["url", "b64_json"] = "url"
    watermark: bool = True


class ImageGenerationResponse(BaseModel):
    """Response model for image generation."""

    created: int
    data: list[dict]  # [{"url": "...", "b64_json": "..."}, ...]


class VolcengineImageClient:
    """Client for Volcengine ARK image generation API."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://ark.cn-beijing.volces.com",
        **kwargs,
    ) -> None:
        """Initialize the Volcengine image client.

        Args:
            api_key: Volcengine ARK API key. If None, uses settings.volcengine_api_key
            base_url: API base URL
            **kwargs: Additional arguments (reserved for future use)
        """
        self._api_key = api_key or settings.volcengine_api_key
        self._base_url = base_url.rstrip("/")
        self._endpoint = f"{self._base_url}/api/v3/images/generations"

    def generate_image(
        self,
        prompt: str,
        model: str = "doubao-seedream-4-0-250828",
        size: str = "2K",
        response_format: Literal["url", "b64_json"] = "url",
        watermark: bool = True,
        timeout: int = 120,
    ) -> str:
        """Generate an image synchronously.

        Args:
            prompt: Text prompt for image generation
            model: Model name to use
            size: Image size - either "1K"/"2K"/"4K" or "WIDTHxHEIGHT" (e.g., "2560x1440")
            response_format: Response format ("url" or "b64_json")
            watermark: Whether to add watermark
            timeout: Request timeout in seconds

        Returns:
            Image URL (if response_format="url") or base64 encoded image (if response_format="b64_json")

        Raises:
            requests.HTTPError: If API request fails
            ValueError: If response format is invalid
        """
        return asyncio.run(
            self.async_generate_image(
                prompt=prompt,
                model=model,
                size=size,
                response_format=response_format,
                watermark=watermark,
                timeout=timeout,
            )
        )

    async def async_generate_image(
        self,
        prompt: str,
        model: str = "doubao-seedream-4-0-250828",
        size: str = "2K",
        response_format: Literal["url", "b64_json"] = "url",
        watermark: bool = True,
        timeout: int = 120,
    ) -> str:
        """Generate an image asynchronously.

        Args:
            prompt: Text prompt for image generation
            model: Model name to use
            size: Image size (1K, 2K, or 4K)
            response_format: Response format ("url" or "b64_json")
            watermark: Whether to add watermark
            timeout: Request timeout in seconds

        Returns:
            Image URL (if response_format="url") or base64 encoded image (if response_format="b64_json")

        Raises:
            requests.HTTPError: If API request fails
            ValueError: If response format is invalid
        """
        request_data = ImageGenerationRequest(
            model=model,
            prompt=prompt,
            size=size,
            response_format=response_format,
            watermark=watermark,
        )

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }

        logger.info(f"Generating image with Volcengine ARK API (model: {model}, size: {size})")
        logger.debug(f"Prompt: {prompt[:100]}...")

        response = requests.post(
            self._endpoint,
            headers=headers,
            json=request_data.model_dump(),
            timeout=timeout,
        )

        response.raise_for_status()

        response_data = ImageGenerationResponse.model_validate(response.json())

        if not response_data.data or len(response_data.data) == 0:
            raise ValueError("No image data in response")

        # Extract URL or b64_json from first image
        image_data = response_data.data[0]
        if response_format == "url":
            if "url" not in image_data:
                raise ValueError("No URL in response data")
            return image_data["url"]
        else:  # b64_json
            if "b64_json" not in image_data:
                raise ValueError("No b64_json in response data")
            return image_data["b64_json"]
