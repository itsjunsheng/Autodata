# Read RefCOCO annotations and import clean images directly into cloud storage.

import json
import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from io import BytesIO
from uuid import NAMESPACE_URL, UUID, uuid5

import httpx
from PIL import Image, ImageDraw

from google_drive import GoogleDrive
from models import Box, SampleRecord, from_coco_xywh
from supabase_client import SupabaseStore

DATASET = "jxu124/refcoco"
ROWS_URL = "https://datasets-server.huggingface.co/rows"
METADATA_URL = f"https://huggingface.co/api/datasets/{DATASET}"
IMAGE_BASE = "https://s3.amazonaws.com/images.cocodataset.org/train2014"
ROW_PAGE_SIZE = 100
MAX_REQUEST_ATTEMPTS = 4
MAX_RETRY_DELAY = 60
RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}


class DatasetError(RuntimeError):
    # A source failure with enough context to identify the failed import step.
    pass


def _retry_after(value: str | None) -> float:
    if not value:
        return 0
    try:
        return max(0, int(value))
    except ValueError:
        try:
            date = parsedate_to_datetime(value)
            return max(0, (date - datetime.now(UTC)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return 0


def request_source(client: httpx.Client, url: str, context: str, *, params=None) -> httpx.Response:
    # Retry temporary source errors; never print response bodies or credentials.
    host = httpx.URL(url).host
    for attempt in range(MAX_REQUEST_ATTEMPTS):
        try:
            response = client.get(url, params=params)
            response.raise_for_status()
            return response
        except httpx.HTTPError as error:
            if isinstance(error, httpx.HTTPStatusError):
                status = error.response.status_code
                reason = f"HTTP {status} from {host}"
                retryable = status in RETRYABLE_STATUS
                retry_after = _retry_after(error.response.headers.get("retry-after"))
            else:
                reason = f"{type(error).__name__} from {host}"
                retryable = isinstance(error, httpx.TransportError)
                retry_after = 0
            message = f"{context}: {reason}."
            if retryable and retry_after > MAX_RETRY_DELAY:
                raise DatasetError(
                    f"{message} The source asks to wait {retry_after:.0f}s. Rerun later; "
                    "saved samples are reused."
                ) from error
            if not retryable or attempt == MAX_REQUEST_ATTEMPTS - 1:
                raise DatasetError(
                    f"{message} Rerun the import; saved samples are reused."
                ) from error
            delay = max(2 ** (attempt + 1), retry_after)
            print(
                f"{message} Retrying in {delay:g}s (request {attempt + 2}/{MAX_REQUEST_ATTEMPTS}).",
                flush=True,
            )
            time.sleep(delay)
    raise AssertionError("Unreachable request state")


def load_image(content: bytes, width: int, height: int) -> Image.Image:
    # Validate original dimensions; keep the annotation's pixel grid unchanged.
    with Image.open(BytesIO(content)) as image:
        image.load()
        if image.size != (width, height):
            raise ValueError("Image dimensions disagree with the grounding annotation.")
        # Do not EXIF-rotate: COCO annotations use the original pixel grid.
        return image.convert("RGB")


def target_views(image: Image.Image, target: Box) -> list[Image.Image]:
    # Privileged original and marked views for the main agent and challenger.
    marked = image.copy()
    draw = ImageDraw.Draw(marked)
    coords = target.pixels(*image.size)
    draw.rectangle(coords, outline="red", width=3)
    draw.text((coords[0] + 3, max(0, coords[1] - 13)), "TARGET", fill="red", stroke_width=1)
    return [image, marked]


@dataclass(frozen=True)
class RefCOCOReference:
    # A source target before its original image is uploaded.

    ref_id: int
    image_id: int
    width: int
    height: int
    target: Box
    referring_expressions: list[str]

    @property
    def sample_id(self) -> UUID:
        return uuid5(NAMESPACE_URL, f"{DATASET}:ref:{self.ref_id}")

    @property
    def image_key(self) -> str:
        return f"coco:train2014:{self.image_id}"

    @property
    def image_url(self) -> str:
        return f"{IMAGE_BASE}/COCO_train2014_{self.image_id:012d}.jpg"

    def sample_record(self, file_id: str) -> SampleRecord:
        return SampleRecord(
            id=self.sample_id,
            image_file_id=file_id,
            width=self.width,
            height=self.height,
            bbox_2d=self.target.as_xyxy(),
            referring_expressions=self.referring_expressions,
            dataset=DATASET,
        )


def reference_from_row(row: dict) -> RefCOCOReference:
    # Use raw COCO XYWH, not the mirror's already-converted top-level bbox.
    annotation = json.loads(row["raw_anns"])
    info = json.loads(row["raw_image_info"])
    if annotation["image_id"] != row["image_id"] or info["id"] != row["image_id"]:
        raise ValueError("RefCOCO image and annotation IDs disagree.")
    if annotation["id"] != row["ann_id"] or annotation.get("iscrowd", 0):
        raise ValueError("Expected a single-object RefCOCO annotation.")
    if row["split"] != "train":
        raise ValueError("Refusing to import evaluation references into the training pool.")
    expressions = [sentence["sent"].strip() for sentence in row["sentences"]]
    if not expressions or any(not expression for expression in expressions):
        raise ValueError("RefCOCO reference has an empty referring expression.")
    return RefCOCOReference(
        ref_id=row["ref_id"],
        image_id=row["image_id"],
        width=info["width"],
        height=info["height"],
        target=from_coco_xywh(annotation["bbox"], info["width"], info["height"]),
        referring_expressions=expressions,
    )


def fetch_rows(client: httpx.Client, offset: int) -> dict:
    response = request_source(
        client,
        ROWS_URL,
        f"RefCOCO annotations at row {offset}",
        params={
            "dataset": DATASET,
            "config": "default",
            "split": "train",
            "offset": offset,
            "length": ROW_PAGE_SIZE,
        },
    )
    payload = response.json()
    expected = min(ROW_PAGE_SIZE, payload["num_rows_total"] - offset)
    if payload.get("partial") or not payload["rows"] or len(payload["rows"]) != expected:
        raise DatasetError(f"The dataset viewer did not return complete rows at offset {offset}.")
    if any(item.get("truncated_cells") for item in payload["rows"]):
        raise DatasetError(f"The dataset viewer truncated annotation fields at offset {offset}.")
    return payload


def iter_references(client: httpx.Client, count: int) -> Iterator[RefCOCOReference]:
    # Read the first distinct images in source order, with one target per image.
    if count < 1:
        raise ValueError("count must be positive")
    first = fetch_rows(client, 0)
    total = first["num_rows_total"]
    if count > total:
        raise ValueError(f"Only {total} references are available.")
    seen = set()
    for offset in range(0, total, ROW_PAGE_SIZE):
        payload = first if offset == 0 else fetch_rows(client, offset)
        if payload["num_rows_total"] != total:
            raise DatasetError("The source row count changed during import. Rerun later.")
        for item in payload["rows"]:
            reference = reference_from_row(item["row"])
            if reference.image_id in seen:
                continue
            seen.add(reference.image_id)
            yield reference
            if len(seen) == count:
                return
    raise ValueError("Not enough distinct images in this split.")


def import_refcoco(drive: GoogleDrive, store: SupabaseStore, count: int) -> dict:
    # Skip complete samples and fill missing images or metadata on a rerun.
    image_ids = set()
    processed = skipped = downloaded = 0
    with httpx.Client(timeout=60, follow_redirects=True) as client:
        metadata = request_source(client, METADATA_URL, "RefCOCO repository metadata")
        revision = metadata.json()["sha"]
        for reference in iter_references(client, count):
            existing = store.get_sample(reference.sample_id)
            if existing and drive.image_exists(existing.image_file_id):
                file_id = existing.image_file_id
                skipped += 1
                action = "Skipped"
            else:
                file_id = drive.find_sample(reference.image_key)
                if file_id is None:
                    response = request_source(
                        client,
                        reference.image_url,
                        f"COCO image {reference.image_id} for reference {reference.ref_id}",
                    )
                    content = response.content
                    load_image(content, reference.width, reference.height)
                    file_id = drive.upload_sample(
                        content, "image/jpeg", source_key=reference.image_key
                    )
                    downloaded += 1
                if existing:
                    store.replace_sample_image(existing.id, file_id)
                else:
                    store.save_sample(reference.sample_record(file_id))
                action = "Imported"
            image_ids.add(file_id)
            processed += 1
            print(f"{action} {processed}/{count}: reference {reference.ref_id}", flush=True)
        latest = request_source(client, METADATA_URL, "RefCOCO repository metadata after import")
        if latest.json()["sha"] != revision:
            raise ValueError(
                "The annotation mirror changed during import. Check it before continuing."
            )
    return {
        "sample_rows": processed,
        "unique_images": len(image_ids),
        "downloaded": downloaded,
        "skipped": skipped,
        "dataset": DATASET,
    }
