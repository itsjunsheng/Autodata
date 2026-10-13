# Call the training-target Qwen model through OpenRouter.

import json
import os

from openai import APIError, OpenAI
from PIL import Image
from pydantic import ValidationError

from model_client import ModelError, data_url
from models import GroundingAnswer, Prediction, Settings
from strong_solver import SOLVER

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class WeakSolver:
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

    def check_connection(self) -> None:
        # Check OpenRouter's catalog before spending generation tokens.
        try:
            models = {model.id: model for model in self.client.models.list().data}
        except APIError as error:
            raise ModelError(
                "Cannot reach OpenRouter. Check OPENROUTER_API_KEY and the network connection."
            ) from error
        model = models.get(self.settings.weak_model)
        if model is None:
            raise ModelError(
                "AUTODATA_WEAK_MODEL must match an OpenRouter model ID, "
                "for example qwen/qwen3-vl-8b-instruct."
            )
        metadata = model.model_extra or {}
        modalities = metadata.get("architecture", {}).get("input_modalities", [])
        if "image" not in modalities:
            raise ModelError("The selected OpenRouter weak model does not support image inputs.")
        if "structured_outputs" not in metadata.get("supported_parameters", []):
            raise ModelError("The selected OpenRouter weak model lacks JSON-schema output support.")

    def solve(self, image: Image.Image, referring_expression: str) -> Prediction:
        # Same clean image, referring expression, and task as the strong solver. No private labels.
        try:
            response = self.client.chat.completions.create(
                model=self.settings.weak_model,
                messages=[
                    {"role": "system", "content": SOLVER},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps({"referring_expression": referring_expression}),
                            },
                            {"type": "image_url", "image_url": {"url": data_url(image)}},
                        ],
                    },
                ],
                max_tokens=self.settings.max_output_tokens,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "grounding_answer",
                        "strict": True,
                        "schema": GroundingAnswer.model_json_schema(),
                    },
                },
                extra_body={"provider": {"require_parameters": True}},
            )
        except APIError as error:
            raise ModelError(
                f"OpenRouter Qwen: {type(error).__name__}; check API key, credits, "
                "model availability and structured-output support."
            ) from error
        if not getattr(response, "choices", None):
            raise ModelError("Qwen returned no choices; no IoU score assigned.")
        choice = response.choices[0]
        if choice.finish_reason != "stop" or not choice.message.content or choice.message.refusal:
            raise ModelError("Qwen returned incomplete/refused output; no IoU score assigned.")
        try:
            return GroundingAnswer.model_validate_json(choice.message.content).as_prediction()
        except ValidationError as error:
            raise ModelError("Qwen returned an invalid box; no IoU score assigned.") from error
