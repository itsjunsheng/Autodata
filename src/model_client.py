# OpenAI and OpenRouter vision requests, pre-flight model checks, and error handling.

import base64
import os
import re
from io import BytesIO
from typing import Protocol, TypeVar

from openai import NOT_GIVEN, APIError, NotFoundError, OpenAI
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

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class ModelClient(Protocol):
    def complete(
        self,
        *,
        role: str,
        model: str,
        instructions: str | None,
        text: str,
        images: list[Image.Image],
        schema: type[T],
    ) -> T: ...

    def check_model(self, model: str, role: str) -> None: ...


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

    def check_model(self, model: str, role: str) -> None:
        # Free metadata lookup; fails before any paid generation call.
        try:
            self.client.models.retrieve(model)
        except NotFoundError as error:
            raise ModelError(
                f"{role}: OpenAI does not offer model {model!r}. Set the role's AUTODATA_*_MODEL "
                "to an OpenAI model ID, or set AUTODATA_STRONG_PROVIDER=openrouter "
                "for an OpenRouter ID."
            ) from error
        except APIError as error:
            raise ModelError(api_error_message(error, role, model)) from error

    def complete(
        self,
        *,
        role: str,
        model: str,
        instructions: str | None,
        text: str,
        images: list[Image.Image],
        schema: type[T],
    ) -> T:
        content = [
            {"type": "input_image", "image_url": data_url(image), "detail": "high"}
            for image in images
        ]
        content.append({"type": "input_text", "text": text})
        try:
            response = self.client.responses.parse(
                model=model,
                instructions=instructions or NOT_GIVEN,
                input=[{"role": "user", "content": content}],
                text_format=schema,
                # Low effort: reasoning tokens bill as output, and these roles need a brief
                # diagnosis, not a long deliberation.
                reasoning={"effort": "low"},
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


class OpenRouterClient:
    def __init__(
        self,
        settings: Settings,
        client: OpenAI | None = None,
    ):
        self.settings = settings
        if client is None:
            # OpenRouter has its own credential; never forward OPENAI_API_KEY.
            key = os.getenv("OPENROUTER_API_KEY", "").strip()
            if not key or key in {"EMPTY", "your-api-key", "your-openrouter-key", "sk-..."}:
                raise ValueError("Set OPENROUTER_API_KEY in .env before running Qwen.")
            client = OpenAI(
                base_url=OPENROUTER_BASE_URL,
                api_key=key,
                timeout=120.0,
                max_retries=2,
            )
        self.client = client
        self._catalogue: dict | None = None

    def check_model(self, model: str, role: str) -> None:
        # Check OpenRouter's catalogue before spending generation tokens.
        if self._catalogue is None:
            try:
                self._catalogue = {entry.id: entry for entry in self.client.models.list().data}
            except APIError as error:
                raise ModelError(
                    f"{role}: cannot reach OpenRouter. "
                    "Check OPENROUTER_API_KEY and the network connection."
                ) from error
        entry = self._catalogue.get(model)
        if entry is None:
            raise ModelError(
                f"{role}: {model!r} must match an OpenRouter model ID, "
                "for example qwen/qwen3-vl-8b-instruct."
            )
        metadata = entry.model_extra or {}
        modalities = metadata.get("architecture", {}).get("input_modalities", [])
        if "image" not in modalities:
            raise ModelError(f"{role}: OpenRouter model {model!r} does not support image inputs.")
        if "structured_outputs" not in metadata.get("supported_parameters", []):
            raise ModelError(
                f"{role}: OpenRouter model {model!r} lacks JSON-schema output support."
            )

    def complete(
        self,
        *,
        role: str,
        model: str,
        instructions: str | None,
        text: str,
        images: list[Image.Image],
        schema: type[T],
    ) -> T:
        content = [{"type": "image_url", "image_url": {"url": data_url(i)}} for i in images]
        content.append({"type": "text", "text": text})
        messages = [{"role": "user", "content": content}]
        if instructions is not None:
            messages.insert(0, {"role": "system", "content": instructions})
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                # Greedy decoding: one solver rollout is reproducible and matches greedy
                # evaluation of the fine-tuned model.
                temperature=0,
                max_tokens=self.settings.max_output_tokens,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema.__name__,
                        "strict": True,
                        "schema": schema.model_json_schema(),
                    },
                },
                extra_body={"provider": {"require_parameters": True}},
            )
        except APIError as error:
            raise ModelError(
                f"{role}: OpenRouter {type(error).__name__}; check API key, credits, "
                "model availability and structured-output support."
            ) from error
        if not getattr(response, "choices", None):
            raise ModelError(f"{role}: OpenRouter returned no choices; no IoU score assigned.")
        choice = response.choices[0]
        if choice.finish_reason != "stop" or not choice.message.content or choice.message.refusal:
            raise ModelError(f"{role}: output was incomplete or refused; no IoU score assigned.")
        try:
            return schema.model_validate_json(choice.message.content)
        except ValidationError as error:
            raise ModelError(
                f"{role}: invalid structured model output; no IoU score assigned."
            ) from error
