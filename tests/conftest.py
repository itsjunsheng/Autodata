from io import BytesIO

import pytest
from PIL import Image

from models import AnnotationRecord, Box, Sample, Settings

TEST_MODELS = {
    "main_model": "gpt-5.6-luna",
    "challenger_model": "gpt-5.6-luna",
    "strong_model": "qwen/qwen3-vl-235b-a22b-instruct",
    "weak_model": "qwen/qwen3-vl-8b-instruct",
}


def make_settings(**overrides) -> Settings:
    return Settings(**(TEST_MODELS | overrides))


@pytest.fixture
def sample() -> Sample:
    return Sample(
        ref_id=1,
        image_file_id="original-image",
        width=200,
        height=100,
        referring_expressions=["the red cup"],
        target=Box(x_min=100, y_min=200, x_max=500, y_max=800),
    )


@pytest.fixture
def image() -> Image.Image:
    return Image.new("RGB", (200, 100), "white")


@pytest.fixture
def image_bytes(image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def sample_record(sample) -> AnnotationRecord:
    return AnnotationRecord(
        image_file_id=sample.image_file_id,
        width=sample.width,
        height=sample.height,
        bbox_2d=sample.target.as_xyxy(),
        referring_expressions=sample.referring_expressions,
        coco_image_id=20,
        ref_id=sample.ref_id,
        category="cup",
    )
