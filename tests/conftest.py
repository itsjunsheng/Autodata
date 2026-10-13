from io import BytesIO
from uuid import UUID

import pytest
from PIL import Image

from models import Box, Sample, SampleRecord


@pytest.fixture
def sample() -> Sample:
    return Sample(
        ref_id=1,
        sample_id=UUID("00000000-0000-0000-0000-000000000001"),
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
def sample_record(sample) -> SampleRecord:
    return SampleRecord(
        id=sample.sample_id,
        image_file_id=sample.image_file_id,
        width=sample.width,
        height=sample.height,
        bbox_2d=sample.target.as_xyxy(),
        referring_expressions=sample.referring_expressions,
        dataset="jxu124/refcoco",
    )
