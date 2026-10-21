# Validated run settings, data structures, and box coordinates.

import math
import os
from collections.abc import Sequence
from typing import Annotated, Literal, Self

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


def text_limit(limit: int) -> AfterValidator:
    # A validator, not a schema keyword: OpenAI strict schemas reject minLength/maxLength.
    def check(value: str) -> str:
        if not 1 <= len(value) <= limit:
            raise ValueError(f"Text must be 1 to {limit} characters.")
        return value

    return AfterValidator(check)


class Box(StrictModel):
    # Normalised XYXY edges: top-left (0, 0), bottom-right (1000, 1000).

    x_min: float = Field(ge=0, le=1000)
    y_min: float = Field(ge=0, le=1000)
    x_max: float = Field(ge=0, le=1000)
    y_max: float = Field(ge=0, le=1000)

    @model_validator(mode="after")
    def positive_area(self) -> Self:
        if self.x_max <= self.x_min or self.y_max <= self.y_min:
            raise ValueError("Box must have positive width and height; do not swap its edges.")
        return self

    def pixels(self, width: int, height: int) -> tuple[float, float, float, float]:
        # Convert 0-1000 coordinates to image pixels for drawing.
        return (
            self.x_min / 1000 * width,
            self.y_min / 1000 * height,
            self.x_max / 1000 * width,
            self.y_max / 1000 * height,
        )

    @classmethod
    def from_pixels(cls, values: Sequence[float], width: int, height: int) -> Self:
        # Normalise pixel XYXY directly to 0-1000.
        if len(values) != 4:
            raise ValueError("Expected four XYXY bounding-box coordinates.")
        if not all(math.isfinite(value) and value > 0 for value in (width, height)):
            raise ValueError("Image width and height must be finite and positive.")
        x1, y1, x2, y2 = values
        return cls.from_xyxy(
            [x1 / width * 1000, y1 / height * 1000, x2 / width * 1000, y2 / height * 1000]
        )

    @classmethod
    def from_xyxy(cls, values: Sequence[float]) -> Self:
        # Validate an already-normalised 0-1000 XYXY box without rescaling it.
        if len(values) != 4:
            raise ValueError("Expected four XYXY bounding-box coordinates.")
        x1, y1, x2, y2 = values
        return cls(x_min=x1, y_min=y1, x_max=x2, y_max=y2)

    def as_xyxy(self, *, round_outward: bool = False) -> list[int] | list[float]:
        # Return the 0-1000 edges; optionally outward-round integer training labels.
        edges = [self.x_min, self.y_min, self.x_max, self.y_max]
        if round_outward:
            return [
                math.floor(edges[0]),
                math.floor(edges[1]),
                math.ceil(edges[2]),
                math.ceil(edges[3]),
            ]
        return edges


def normalize_bbox(
    values: Sequence[float], width: int, height: int, *, round_outward: bool = False
) -> list[int] | list[float]:
    # Convert pixel XYXY directly to 0-1000 with one call.
    #
    # Example: normalize_bbox([80, 80, 400, 320], width=800, height=400)
    # returns [100, 200, 500, 800]. Use round_outward=True for integer training labels.
    return Box.from_pixels(values, width, height).as_xyxy(round_outward=round_outward)


class Sample(StrictModel):
    # One database target passed to the agent loop.

    ref_id: int = Field(ge=0)
    image_file_id: str = Field(min_length=1)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    referring_expressions: list[Annotated[str, Field(min_length=1)]] = Field(min_length=1)
    target: Box


class Candidate(StrictModel):
    referring_expression: Annotated[str, text_limit(400)]


class ChallengerDirection(StrictModel):
    # The main LLM's brief assessment and instructions for the next candidate.

    assessment: Annotated[str, text_limit(1000)]
    challenger_prompt: Annotated[str, text_limit(1600)]


class Claim(StrictModel):
    claim: Annotated[str, text_limit(200)]
    visible: bool


class Verdict(StrictModel):
    # The judge's review of one referring expression against the marked target.

    claims: list[Claim]
    other_matches: int = Field(ge=0)
    reason: Annotated[str, text_limit(400)]


class Prediction(StrictModel):
    bbox: Box


class GroundingAnswer(StrictModel):
    # Shared OpenAI/Qwen response, following Qwen3-VL's 0-1000 XYXY convention.

    bbox_2d: list[Annotated[float, Field(ge=0, le=1000)]] = Field(min_length=4, max_length=4)

    @model_validator(mode="after")
    def valid_box(self) -> Self:
        self.as_prediction()
        return self

    def as_prediction(self) -> Prediction:
        return Prediction(bbox=Box.from_xyxy(self.bbox_2d))


class AnnotationRecord(GroundingAnswer):
    # One annotated object; several objects may share the same Drive image.

    image_file_id: str = Field(min_length=1)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    referring_expressions: list[Annotated[str, Field(min_length=1)]] = Field(min_length=1)
    coco_image_id: int = Field(gt=0)
    ref_id: int = Field(ge=0)
    category: str = Field(min_length=1)

    def as_sample(self) -> Sample:
        return Sample(
            ref_id=self.ref_id,
            image_file_id=self.image_file_id,
            width=self.width,
            height=self.height,
            referring_expressions=self.referring_expressions,
            target=self.as_prediction().bbox,
        )


class AcceptedExample(GroundingAnswer):
    # A training example: an accepted round joined to its annotation.

    image_file_id: str = Field(min_length=1)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    referring_expression: str = Field(min_length=1, max_length=400)

    @classmethod
    def from_round_row(cls, row: dict) -> "AcceptedExample":
        return cls.model_validate(
            {"referring_expression": row["referring_expression"], **row["refcoco_annotations"]}
        )


class Round(StrictModel):
    number: int
    direction: ChallengerDirection | None = None
    candidate: Candidate | None = None
    weak: Prediction | None = None
    strong: Prediction | None = None
    weak_iou: float | None = None
    strong_iou: float | None = None
    judge: Verdict | None = None
    accepted: bool = False
    feedback: str = ""


class Result(StrictModel):
    sample: Sample
    status: Literal["running", "accepted", "exhausted", "error"] = "running"
    rounds: list[Round] = Field(default_factory=list)
    error: str | None = None


class Settings(StrictModel):
    # Model IDs have no defaults in code; `run` reads all five from .env.
    main_model: str = Field(min_length=1)
    challenger_model: str = Field(min_length=1)
    strong_provider: Literal["openai", "openrouter"] = "openrouter"
    strong_model: str = Field(min_length=1)
    weak_model: str = Field(min_length=1)
    judge_model: str = Field(min_length=1)
    max_rounds: int = Field(default=6, ge=1, le=20)
    iou_threshold: float = Field(default=0.5, gt=0, le=1)
    min_iou_gap: float = Field(default=0.2, ge=0, le=1)
    # Reasoning tokens count toward this cap; an exhausted cap is an incomplete response.
    max_output_tokens: int = Field(default=8000, ge=256)

    @classmethod
    def from_env(cls, **overrides: object) -> "Settings":
        models = {}
        missing = []
        for role in ("main", "challenger", "strong", "weak", "judge"):
            name = f"AUTODATA_{role.upper()}_MODEL"
            value = os.getenv(name, "").strip()
            if value:
                models[f"{role}_model"] = value
            else:
                missing.append(name)
        if missing:
            raise ValueError(f"Set {', '.join(missing)} in .env.")
        provider = os.getenv("AUTODATA_STRONG_PROVIDER", "").strip().lower()
        if provider:
            models["strong_provider"] = provider
        return cls(**(models | overrides))


def from_coco_xywh(values: Sequence[float], width: int, height: int) -> Box:
    # Convert COCO pixel [x, y, width, height] to normalised XYXY.
    if len(values) != 4 or width <= 0 or height <= 0:
        raise ValueError("Expected four COCO coordinates and positive image dimensions.")
    x, y, w, h = values
    if not all(math.isfinite(v) for v in values) or w <= 0 or h <= 0:
        raise ValueError("COCO box must be finite with positive width and height.")
    # Allow only small annotation rounding at the image boundary.
    if x < -0.01 or y < -0.01 or x + w > width + 0.01 or y + h > height + 0.01:
        raise ValueError("COCO box lies outside the original image.")
    return Box.from_pixels(
        [max(0, x), max(0, y), min(width, x + w), min(height, y + h)],
        width,
        height,
    )
