# OpenAI vision requests and model-call error handling.

import base64
import json
import os
import re
from io import BytesIO
from typing import TypeVar

from openai import APIError, OpenAI
from PIL import Image
from pydantic import BaseModel, ValidationError

from models import Settings


class ModelError(RuntimeError):
    # A failed model call, never evidence that a generated example is difficult.
    pass


def api_error_message(error: APIError, role: str, model: str) -> str:
    # Keep useful quota diagnostics without copying arbitrary server error bodies.
    status = getattr(error, "status_code", None)
    parts = [f"{role}: {type(error).__name__} (HTTP {status}); model={model}"]
    code = error.code
    if code in {
        "rate_limit_exceeded",
        "insufficient_quota",
        "billing_hard_limit_reached",
        "slow_down",
        "server_is_overloaded",
    }:
        parts.append(f"code={code}")
    response = getattr(error, "response", None)
    if response is not None:
        for name in (
            "retry-after",
            "retry-after-ms",
            "x-ratelimit-limit-requests",
            "x-ratelimit-remaining-requests",
            "x-ratelimit-reset-requests",
            "x-ratelimit-limit-tokens",
            "x-ratelimit-remaining-tokens",
            "x-ratelimit-reset-tokens",
            "x-ratelimit-limit-project-tokens",
            "x-ratelimit-remaining-project-tokens",
            "x-ratelimit-reset-project-tokens",
        ):
            value = response.headers.get(name, "")
            if re.fullmatch(r"[0-9][0-9.dhms]{0,39}", value):
                parts.append(f"{name}={value}")
    if code in {"insufficient_quota", "billing_hard_limit_reached"}:
        parts.append("Check API credits and spending limits; waiting does not fix quota errors.")
    elif status == 429:
        parts.append("Check request/token limits and billing; follow Retry-After when present.")
    else:
        parts.append("Check API access.")
    return "; ".join(parts)


def data_url(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


T = TypeVar("T", bound=BaseModel)


class OpenAIClient:
    def __init__(
        self,
        settings: Settings,
        client: OpenAI | None = None,
    ):
        self.settings = settings
        if client is None:
            key = os.getenv("OPENAI_API_KEY", "").strip()
            if not key or key in {"your-api-key", "sk-...", "YOUR_API_KEY"}:
                raise ValueError("Set OPENAI_API_KEY in .env before running generation.")
            client = OpenAI(api_key=key, timeout=90.0, max_retries=2)
        self.client = client

    def complete(
        self,
        *,
        role: str,
        model: str,
        instructions: str,
        payload: dict,
        images: list[Image.Image],
        schema: type[T],
    ) -> T:
        content = [{"type": "input_text", "text": json.dumps(payload, ensure_ascii=False)}]
        content.extend(
            {"type": "input_image", "image_url": data_url(image), "detail": "high"}
            for image in images
        )
        try:
            response = self.client.responses.parse(
                model=model,
                instructions=instructions,
                input=[{"role": "user", "content": content}],
                text_format=schema,
                max_output_tokens=self.settings.max_output_tokens,
                store=False,
            )
        except APIError as error:
            raise ModelError(api_error_message(error, role, model)) from error
        except (ValidationError, ValueError) as error:
            raise ModelError(
                f"{role}: invalid structured model output; no IoU score assigned."
            ) from error

        if response.status != "completed" or response.output_parsed is None:
            raise ModelError(f"{role}: response was incomplete, refused, or had no parsed output.")
        return response.output_parsed
