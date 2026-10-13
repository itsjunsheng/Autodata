# Render attempt previews in memory and explicitly export Qwen training data.

import json
import textwrap
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from dataset import load_image
from google_drive import GoogleDrive, validate_file_id
from models import OutputRecord, Result, Round, Sample
from supabase_client import SupabaseStore


def attempt_status(result: Result, attempt: Round) -> str:
    if result.status == "error" and attempt is result.rounds[-1]:
        return "error"
    if attempt.accepted:
        return "accepted"
    return "rejected" if attempt.feedback else "running"


def render_attempt_image(
    image: Image.Image, sample: Sample, attempt: Round, status: str
) -> Image.Image:
    # The added text and coloured boxes are inspection aids, never training inputs.
    marked = image.copy()
    draw = ImageDraw.Draw(marked)
    # Different line widths keep all three colours visible when boxes coincide.
    boxes = [(sample.target, "lime", 7)]
    if attempt.weak:
        boxes.append((attempt.weak.bbox, "red", 5))
    if attempt.strong:
        boxes.append((attempt.strong.bbox, "blue", 3))
    for box, colour, width in boxes:
        draw.rectangle(box.pixels(*image.size), outline=colour, width=width)

    font = ImageFont.load_default(size=14)
    legends = [
        ("lime", "Ground"),
        ("red", "Weak solver" + (" (unavailable)" if attempt.weak is None else "")),
        ("blue", "Strong solver" + (" (unavailable)" if attempt.strong is None else "")),
    ]
    legend_width = sum(20 + font.getlength(label) + 24 for _, label in legends)
    canvas_width = max(image.width, 400, int(legend_width + font.getlength(status.upper())) + 24)
    referring_expression = (
        attempt.candidate.referring_expression
        if attempt.candidate
        else "No referring expression generated."
    )
    line_width = (canvas_width - 24) // 8
    lines = textwrap.wrap(referring_expression, width=line_width)
    while any(font.getlength(line) > canvas_width - 24 for line in lines):
        line_width -= 1
        lines = textwrap.wrap(referring_expression, width=line_width)
    header_height = 44 + 20 * len(lines)
    canvas = Image.new("RGB", (canvas_width, image.height + header_height), "white")
    canvas.paste(marked, ((canvas_width - image.width) // 2, header_height))
    draw = ImageDraw.Draw(canvas)
    for index, line in enumerate(lines):
        draw.text((12, 12 + 20 * index), line, fill="black", font=font)
    legend_y = 20 + 20 * len(lines)
    x = 12
    for colour, label in legends:
        draw.rectangle((x, legend_y + 3, x + 12, legend_y + 13), fill=colour)
        draw.text((x + 20, legend_y), label, fill="black", font=font)
        x += 20 + font.getlength(label) + 24
    draw.text((x, legend_y), status.upper(), fill="black", font=font)
    return canvas


def _training_record(example: OutputRecord, image_path: str) -> dict:
    return {
        "images": [image_path],
        "conversations": [
            {
                "from": "human",
                "value": (
                    "<image>\nLocate the object described by the referring expression: "
                    f"{json.dumps(example.referring_expression)}. "
                    'Return JSON {"bbox_2d": [x1, y1, x2, y2]} using XYXY coordinates '
                    "normalised to 0-1000 relative to the full image."
                ),
            },
            {
                "from": "gpt",
                "value": json.dumps(
                    {"bbox_2d": example.as_prediction().bbox.as_xyxy(round_outward=True)}
                ),
            },
        ],
    }


def export_training(store: SupabaseStore, drive: GoogleDrive, run_id: str, output: Path) -> int:
    # An explicit opt-in download for training; generation never writes local files.
    if output.exists() and any(output.iterdir()):
        raise ValueError("Training output directory is not empty; choose another --output.")
    records, downloaded = [], {}
    for example in store.iter_outputs(run_id):
        file_id = validate_file_id(example.image_file_id)
        if file_id not in downloaded:
            content = drive.download(file_id)
            load_image(content, example.width, example.height)
            with Image.open(BytesIO(content)) as image:
                suffix = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}.get(image.format)
            if suffix is None:
                raise ValueError("Training images must be JPEG, PNG, or WebP.")
            relative = f"images/{file_id}{suffix}"
            destination = output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
            downloaded[file_id] = (relative, example.width, example.height)
        relative, width, height = downloaded[file_id]
        if (width, height) != (example.width, example.height):
            raise ValueError("Accepted records disagree about their shared image dimensions.")
        records.append(_training_record(example, relative))
    if not records:
        raise ValueError("No accepted examples to export for this run.")
    (output / "qwen_train.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "dataset_info.json").write_text(
        json.dumps(
            {
                "autodata_grounding": {
                    "file_name": "qwen_train.json",
                    "formatting": "sharegpt",
                    "columns": {"messages": "conversations", "images": "images"},
                    "tags": {
                        "role_tag": "from",
                        "content_tag": "value",
                        "user_tag": "human",
                        "assistant_tag": "gpt",
                    },
                }
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return len(records)
