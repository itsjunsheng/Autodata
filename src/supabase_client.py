# Store grounding metadata, agent rounds, and run settings in Supabase.

import os
from collections.abc import Iterator
from datetime import UTC, datetime
from urllib.parse import urlsplit
from uuid import uuid4

import httpx
from postgrest.exceptions import APIError
from supabase import Client, ClientOptions, create_client

from models import AcceptedExample, AnnotationRecord, Result, Sample, Settings

ANNOTATION_FIELDS = (
    "ref_id,image_file_id,width,height,bbox_2d,referring_expressions,coco_image_id,category"
)
ACCEPTED_FIELDS = "referring_expression,refcoco_annotations(image_file_id,width,height,bbox_2d)"
PAGE_SIZE = 50


class SupabaseError(RuntimeError):
    # A database failure with a message safe to print.
    pass


def create_supabase_client() -> Client:
    url = os.getenv("SUPABASE_URL", "").strip().rstrip("/")
    key = os.getenv("SUPABASE_API_KEY", "").strip()
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.path
        or parsed.query
        or parsed.fragment
        or parsed.username
        or parsed.password
    ):
        raise SupabaseError("Set SUPABASE_URL to the project's HTTPS API URL.")
    if not key or key in {"your-api-key", "your-supabase-key"} or key.startswith("sb_publishable_"):
        raise SupabaseError("Set SUPABASE_API_KEY to your backend secret/service-role key.")
    return create_client(
        url,
        key,
        options=ClientOptions(
            auto_refresh_token=False,
            persist_session=False,
            postgrest_client_timeout=20,
        ),
    )


def _execute(query, operation: str):
    try:
        return query.execute()
    except (APIError, httpx.HTTPError) as error:
        raise SupabaseError(
            f"Supabase could not {operation}; check access and connectivity."
        ) from error


def check_supabase_connection(client: Client) -> dict:
    # Read-only checks of the required tables and columns; no Storage bucket.
    columns = {
        "refcoco_annotations": ANNOTATION_FIELDS,
        "runs": (
            "id,main_model,challenger_model,weak_model,strong_model,judge_model,"
            "iou_threshold,min_iou_gap,max_rounds,status,finished_at"
        ),
        "rounds": (
            "run_id,ref_id,round_number,referring_expression,weak_bbox_2d,"
            "strong_bbox_2d,weak_iou,strong_iou,status,feedback,preview_file_id"
        ),
    }
    issues = []
    for name, fields in columns.items():
        try:
            client.table(name).select(fields).limit(0).execute()
        except APIError as error:
            if error.code in {"PGRST205", "42P01", "PGRST204", "42703"}:
                issues.append(name)
                continue
            raise SupabaseError(
                f"Supabase access failed for {name}; check the backend key."
            ) from error
        except httpx.HTTPError as error:
            raise SupabaseError("Supabase connection failed; check connectivity.") from error
    return {"connected": True, "ready": not issues, "schema_issues": issues}


class SupabaseStore:
    # Read source targets and persist workflow records; image bytes live in Drive.

    def __init__(self, client: Client):
        self.client = client
        self.run_id = str(uuid4())
        self.saved_rounds: dict[tuple[int, int], dict] = {}

    def save_annotation(self, record: AnnotationRecord) -> None:
        _execute(
            self.client.table("refcoco_annotations").upsert(
                record.model_dump(mode="json"), on_conflict="ref_id", ignore_duplicates=True
            ),
            "save the annotation",
        )

    def get_annotation(self, ref_id: int) -> AnnotationRecord | None:
        rows = _execute(
            self.client.table("refcoco_annotations")
            .select(ANNOTATION_FIELDS)
            .eq("ref_id", ref_id)
            .limit(1),
            "look up the annotation",
        ).data
        return AnnotationRecord.model_validate(rows[0]) if rows else None

    def replace_annotation_image(self, ref_id: int, file_id: str) -> None:
        # Repair a missing image link while preserving the annotation's labels.
        rows = _execute(
            self.client.table("refcoco_annotations")
            .update({"image_file_id": file_id})
            .eq("ref_id", ref_id),
            "repair the annotation image link",
        ).data
        if not rows:
            raise SupabaseError("The annotation was removed during import. Rerun the command.")

    def annotation_count(self) -> int:
        query = self.client.table("refcoco_annotations").select("ref_id", count="exact")
        response = _execute(query.limit(0), "count samples")
        if response.count is None:
            raise SupabaseError("Supabase did not return the annotation count.")
        return response.count

    def iter_samples(self, limit: int, *, offset: int = 0) -> Iterator[Sample]:
        # Read bounded pages in import order; ref_id breaks timestamp ties.
        if limit < 1 or offset < 0:
            raise ValueError("limit must be positive and offset must be non-negative")
        loaded = 0
        while loaded < limit:
            size = min(PAGE_SIZE, limit - loaded)
            rows = _execute(
                self.client.table("refcoco_annotations")
                .select(ANNOTATION_FIELDS)
                .order("created_at")
                .order("ref_id")
                .range(offset + loaded, offset + loaded + size - 1),
                "read samples",
            ).data
            if not rows:
                raise SupabaseError("Samples changed during the run. Start a new batch.")
            for row in rows:
                loaded += 1
                yield AnnotationRecord.model_validate(row).as_sample()

    def start(self, settings: Settings) -> None:
        _execute(
            self.client.table("runs").insert(
                {
                    "id": self.run_id,
                    **settings.model_dump(
                        include={
                            "main_model",
                            "challenger_model",
                            "weak_model",
                            "strong_model",
                            "judge_model",
                            "iou_threshold",
                            "min_iou_gap",
                            "max_rounds",
                        }
                    ),
                    "status": "running",
                }
            ),
            "start the run",
        )

    def save_round(self, result: Result, status: str, preview_file_id: str | None = None) -> None:
        # Checkpoint a loop; an accepted round is the accepted example.
        current = result.rounds[-1]
        key = (result.sample.ref_id, current.number)
        row = {
            "run_id": self.run_id,
            "ref_id": result.sample.ref_id,
            "round_number": current.number,
            "referring_expression": current.candidate.referring_expression
            if current.candidate
            else None,
            "weak_bbox_2d": current.weak.bbox.as_xyxy() if current.weak else None,
            "strong_bbox_2d": current.strong.bbox.as_xyxy() if current.strong else None,
            "weak_iou": current.weak_iou,
            "strong_iou": current.strong_iou,
            "status": status,
            "feedback": result.error if status == "error" else current.feedback,
            "preview_file_id": preview_file_id,
        }
        if self.saved_rounds.get(key) != row:
            _execute(
                self.client.table("rounds").upsert(row, on_conflict="run_id,ref_id,round_number"),
                "save the round",
            )
            self.saved_rounds[key] = row

    def finish(self, status: str) -> None:
        _execute(
            self.client.table("runs")
            .update({"status": status, "finished_at": datetime.now(UTC).isoformat()})
            .eq("id", self.run_id),
            "finish the run",
        )

    def iter_accepted_examples(self, run_ids: list[str]) -> Iterator[AcceptedExample]:
        # Accepted rounds of the given runs, joined to their annotation for image, size and box.
        offset = 0
        while True:
            rows = _execute(
                self.client.table("rounds")
                .select(ACCEPTED_FIELDS)
                .in_("run_id", run_ids)
                .eq("status", "accepted")
                .order("run_id")
                .order("ref_id")
                .order("round_number")
                .range(offset, offset + PAGE_SIZE - 1),
                "read accepted examples",
            ).data
            for row in rows:
                yield AcceptedExample.from_round_row(row)
            if len(rows) < PAGE_SIZE:
                return
            offset += len(rows)
