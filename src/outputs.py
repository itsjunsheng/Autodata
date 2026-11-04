# Render accepted rounds as PNGs in memory and explicitly export Qwen training data.

import json
import textwrap
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from dataset import load_image
from google_drive import GoogleDrive, validate_file_id
from models import AcceptedExample, Result, Round, Sample
from solver import grounding_prompt
from supabase_client import SupabaseStore


def round_status(result: Result, current: Round) -> str:
    if result.status == "error" and current is result.rounds[-1]:
        return "error"
    if current.accepted:
        return "accepted"
    return "rejected" if current.feedback else "running"


def render_accepted(image: Image.Image, sample: Sample, current: Round) -> bytes:
    # A PNG of an accepted round: its referring expression above the image with the ground
    # truth, weak and strong boxes drawn on. An inspection aid, never a training input.
    marked = image.copy()
    draw = ImageDraw.Draw(marked)
    # Different line widths keep all three colours visible when boxes coincide.
    boxes = [
        (sample.target, "lime", 7, "Ground"),
        (current.weak.bbox, "red", 5, "Weak solver"),
        (current.strong.bbox, "blue", 3, "Strong solver"),
    ]
    for box, colour, width, _ in boxes:
        draw.rectangle(box.pixels(*image.size), outline=colour, width=width)

    font = ImageFont.load_default(size=14)
    legend_width = sum(20 + font.getlength(label) + 24 for _, _, _, label in boxes)
    canvas_width = max(image.width, 400, int(legend_width) + 24)
    referring_expression = current.candidate.referring_expression
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
    for _, colour, _, label in boxes:
        draw.rectangle((x, legend_y + 3, x + 12, legend_y + 13), fill=colour)
        draw.text((x + 20, legend_y), label, fill="black", font=font)
        x += 20 + font.getlength(label) + 24
    buffer = BytesIO()
    canvas.save(buffer, format="PNG")
    return buffer.getvalue()


def _training_record(example: AcceptedExample, image_path: str) -> dict:
    return {
        "images": [image_path],
        "conversations": [
            {
                "from": "human",
                "value": "<image>\n" + grounding_prompt(example.referring_expression),
            },
            {
                "from": "gpt",
                "value": json.dumps(
                    {"bbox_2d": example.as_prediction().bbox.as_xyxy(round_outward=True)}
                ),
            },
        ],
    }


def export_training(
    store: SupabaseStore, drive: GoogleDrive, run_ids: list[str], output: Path
) -> int:
    # An explicit opt-in download for training; generation never writes local files.
    if output.exists() and any(output.iterdir()):
        raise ValueError("Training output directory is not empty; choose another --output.")
    records, downloaded = [], {}
    for example in store.iter_accepted_examples(run_ids):
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
