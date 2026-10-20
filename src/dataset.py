# Read RefCOCO annotations and import clean images directly into cloud storage.

import argparse
import json
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from io import BytesIO
from pathlib import Path

import httpx
from dotenv import load_dotenv
from PIL import Image, ImageDraw

from google_drive import DriveError, GoogleDrive, create_drive_client
from models import AnnotationRecord, Box, from_coco_xywh
from supabase_client import (
    SupabaseError,
    SupabaseStore,
    check_supabase_connection,
    create_supabase_client,
)

DATASET = "jxu124/refcoco"
ROWS_URL = "https://datasets-server.huggingface.co/rows"
METADATA_URL = f"https://huggingface.co/api/datasets/{DATASET}"
IMAGE_BASE = "https://s3.amazonaws.com/images.cocodataset.org/train2014"
ROW_PAGE_SIZE = 100
MAX_REQUEST_ATTEMPTS = 4
MAX_RETRY_DELAY = 60
RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}
COCO_CATEGORIES: dict[int, str] = {
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    5: "airplane",
    6: "bus",
    7: "train",
    8: "truck",
    9: "boat",
    10: "traffic light",
    11: "fire hydrant",
    13: "stop sign",
    14: "parking meter",
    15: "bench",
    16: "bird",
    17: "cat",
    18: "dog",
    19: "horse",
    20: "sheep",
    21: "cow",
    22: "elephant",
    23: "bear",
    24: "zebra",
    25: "giraffe",
    27: "backpack",
    28: "umbrella",
    31: "handbag",
    32: "tie",
    33: "suitcase",
    34: "frisbee",
    35: "skis",
    36: "snowboard",
    37: "sports ball",
    38: "kite",
    39: "baseball bat",
    40: "baseball glove",
    41: "skateboard",
    42: "surfboard",
    43: "tennis racket",
    44: "bottle",
    46: "wine glass",
    47: "cup",
    48: "fork",
    49: "knife",
    50: "spoon",
    51: "bowl",
    52: "banana",
    53: "apple",
    54: "sandwich",
    55: "orange",
    56: "broccoli",
    57: "carrot",
    58: "hot dog",
    59: "pizza",
    60: "donut",
    61: "cake",
    62: "chair",
    63: "couch",
    64: "potted plant",
    65: "bed",
    67: "dining table",
    70: "toilet",
    72: "tv",
    73: "laptop",
    74: "mouse",
    75: "remote",
    76: "keyboard",
    77: "cell phone",
    78: "microwave",
    79: "oven",
    80: "toaster",
    81: "sink",
    82: "refrigerator",
    84: "book",
    85: "clock",
    86: "vase",
    87: "scissors",
    88: "teddy bear",
    89: "hair drier",
    90: "toothbrush",
}


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
    category: str
    width: int
    height: int
    target: Box
    referring_expressions: list[str]

    @property
    def file_name(self) -> str:
        # Drive name: the bare COCO image ID, which is the coco_image_id column.
        return f"{self.image_id}.jpg"

    @property
    def image_url(self) -> str:
        return f"{IMAGE_BASE}/COCO_train2014_{self.image_id:012d}.jpg"

    def sample_record(self, file_id: str) -> AnnotationRecord:
        return AnnotationRecord(
            image_file_id=file_id,
            width=self.width,
            height=self.height,
            bbox_2d=self.target.as_xyxy(),
            referring_expressions=self.referring_expressions,
            coco_image_id=self.image_id,
            ref_id=self.ref_id,
            category=self.category,
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
    category_id = annotation["category_id"]
    if category_id != row["category_id"]:
        raise ValueError("RefCOCO category IDs disagree.")
    if category_id not in COCO_CATEGORIES:
        raise ValueError(f"Unknown COCO category ID {category_id}.")
    expressions = [sentence["sent"].strip() for sentence in row["sentences"]]
    if not expressions or any(not expression for expression in expressions):
        raise ValueError("RefCOCO reference has an empty referring expression.")
    return RefCOCOReference(
        ref_id=row["ref_id"],
        image_id=row["image_id"],
        category=COCO_CATEGORIES[category_id],
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
    # Read every referred object of the first `count` images, in source order.
    if count < 1:
        raise ValueError("count must be positive")
    first = fetch_rows(client, 0)
    total = first["num_rows_total"]
    if count > total:
        raise ValueError(f"Only {total} references are available.")
    seen = set()
    current = None
    for offset in range(0, total, ROW_PAGE_SIZE):
        payload = first if offset == 0 else fetch_rows(client, offset)
        if payload["num_rows_total"] != total:
            raise DatasetError("The source row count changed during import. Rerun later.")
        for item in payload["rows"]:
            reference = reference_from_row(item["row"])
            if reference.image_id != current:
                # Rows of one image are adjacent: a new image ID completes the previous one.
                if reference.image_id in seen:
                    raise DatasetError(
                        "RefCOCO rows for one image are not adjacent. Check the source."
                    )
                if len(seen) == count:
                    return
                seen.add(reference.image_id)
                current = reference.image_id
            yield reference
    if len(seen) < count:
        raise ValueError("Not enough distinct images in this split.")


def import_refcoco(drive: GoogleDrive, store: SupabaseStore, count: int) -> dict:
    # Import every referred object of the first `count` images. Drive is checked by file name
    # first, then the row is created, repaired or skipped.
    images: dict[int, str] = {}  # COCO image ID -> Drive file ID, shared by its objects
    objects = skipped = downloaded = 0
    with httpx.Client(timeout=60, follow_redirects=True) as client:
        metadata = request_source(client, METADATA_URL, "RefCOCO repository metadata")
        revision = metadata.json()["sha"]
        for reference in iter_references(client, count):
            file_id = images.get(reference.image_id) or drive.find_image(reference.file_name)
            if file_id is None:
                response = request_source(
                    client,
                    reference.image_url,
                    f"COCO image {reference.image_id} for reference {reference.ref_id}",
                )
                content = response.content
                load_image(content, reference.width, reference.height)
                file_id = drive.upload_image(reference.file_name, content, "image/jpeg")
                downloaded += 1
            images[reference.image_id] = file_id
            existing = store.get_annotation(reference.ref_id)
            if existing is None:
                store.save_annotation(reference.sample_record(file_id))
                action = "Imported"
            elif existing.image_file_id != file_id:
                store.replace_annotation_image(reference.ref_id, file_id)
                action = "Repaired"
            else:
                skipped += 1
                action = "Skipped"
            objects += 1
            print(f"{action} reference {reference.ref_id}, image {len(images)}/{count}", flush=True)
        latest = request_source(client, METADATA_URL, "RefCOCO repository metadata after import")
        if latest.json()["sha"] != revision:
            raise ValueError(
                "The annotation mirror changed during import. Check it before continuing."
            )
    return {
        "sample_rows": objects,
        "unique_images": len(images),
        "downloaded": downloaded,
        "skipped": skipped,
    }


def positive_int(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def main(argv: list[str] | None = None) -> int:
    root = argparse.ArgumentParser(description="Import RefCOCO into Google Drive and Supabase.")
    root.add_argument(
        "--count",
        type=positive_int,
        default=10,
        help="Number of source images; every referred object in them is imported.",
    )
    args = root.parse_args(argv)
    load_dotenv(Path.cwd() / ".env", override=False)
    try:
        client = create_supabase_client()
        if not check_supabase_connection(client)["ready"]:
            raise SupabaseError(
                "Supabase is not ready. Run the supabase/ SQL files in order in the SQL Editor."
            )
        drive = create_drive_client()
        drive.check_connection()
        summary = import_refcoco(drive, SupabaseStore(client), args.count)
        print(json.dumps(summary, indent=2))
        return 0
    except (
        OSError,
        ValueError,
        KeyError,
        httpx.HTTPError,
        DatasetError,
        SupabaseError,
        DriveError,
    ) as error:
        # Provider wrappers intentionally avoid printing credentials or raw server bodies.
        message = (
            "Dataset request failed; check the source endpoint and connectivity."
            if isinstance(error, httpx.HTTPError)
            else str(error)
        )
        print(f"Error: {message}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
