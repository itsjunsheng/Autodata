# Store grounding metadata, agent attempts, and run settings in Supabase.

import os
from collections.abc import Iterator
from datetime import UTC, datetime
from urllib.parse import urlsplit
from uuid import UUID, uuid4, uuid5

import httpx
from postgrest.exceptions import APIError
from supabase import Client, ClientOptions, create_client

from models import OutputRecord, Result, Sample, SampleRecord, Settings

SAMPLE_FIELDS = "id,image_file_id,width,height,bbox_2d,referring_expressions,dataset"
OUTPUT_FIELDS = "attempt_id,image_file_id,width,height,referring_expression,bbox_2d"
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
        "samples": SAMPLE_FIELDS,
        "runs": (
            "id,main_model,challenger_model,weak_model,strong_model,acceptance,"
            "iou_threshold,min_iou_gap,max_rounds,status,finished_at"
        ),
        "attempts": (
            "id,run_id,sample_id,attempt_number,referring_expression,weak_bbox_2d,"
            "strong_bbox_2d,weak_iou,strong_iou,status,feedback,attempt_file_id"
        ),
        "outputs": OUTPUT_FIELDS,
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
        self.saved_attempts: dict[str, dict] = {}

    def save_sample(self, record: SampleRecord) -> None:
        _execute(
            self.client.table("samples").upsert(
                record.model_dump(mode="json"), on_conflict="id", ignore_duplicates=True
            ),
            "save the sample",
        )

    def get_sample(self, sample_id: UUID) -> SampleRecord | None:
        rows = _execute(
            self.client.table("samples").select(SAMPLE_FIELDS).eq("id", str(sample_id)).limit(1),
            "look up the sample",
        ).data
        return SampleRecord.model_validate(rows[0]) if rows else None

    def replace_sample_image(self, sample_id: UUID, file_id: str) -> None:
        # Repair a missing image link while preserving the sample's labels.
        rows = _execute(
            self.client.table("samples")
            .update({"image_file_id": file_id})
            .eq("id", str(sample_id)),
            "repair the sample image link",
        ).data
        if not rows:
            raise SupabaseError("The sample was removed during import. Rerun the command.")

    def _sample_query(self, dataset: str | None):
        query = self.client.table("samples").select(SAMPLE_FIELDS)
        return query.eq("dataset", dataset) if dataset else query

    def sample_count(self, dataset: str | None = None) -> int:
        query = self.client.table("samples").select("id", count="exact")
        if dataset:
            query = query.eq("dataset", dataset)
        response = _execute(query.limit(0), "count samples")
        if response.count is None:
            raise SupabaseError("Supabase did not return the sample count.")
        return response.count

    def iter_samples(
        self, limit: int, *, offset: int = 0, dataset: str | None = None
    ) -> Iterator[Sample]:
        # Read bounded pages in import order; UUIDs break timestamp ties.
        if limit < 1 or offset < 0:
            raise ValueError("limit must be positive and offset must be non-negative")
        loaded = 0
        while loaded < limit:
            size = min(PAGE_SIZE, limit - loaded)
            rows = _execute(
                self._sample_query(dataset)
                .order("created_at")
                .order("id")
                .range(offset + loaded, offset + loaded + size - 1),
                "read samples",
            ).data
            if not rows:
                raise SupabaseError("Samples changed during the run. Start a new batch.")
            for row in rows:
                loaded += 1
                yield SampleRecord.model_validate(row).as_sample(offset + loaded)

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
                            "acceptance",
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

    def save_attempt(self, result: Result, status: str, attempt_file_id: str | None = None) -> None:
        # Checkpoint a loop; the database also inserts its output when accepted.
        attempt = result.rounds[-1]
        identifier = str(uuid5(UUID(self.run_id), f"{result.sample.sample_id}:{attempt.number}"))
        row = {
            "id": identifier,
            "run_id": self.run_id,
            "sample_id": str(result.sample.sample_id),
            "attempt_number": attempt.number,
            "referring_expression": attempt.candidate.referring_expression
            if attempt.candidate
            else None,
            "weak_bbox_2d": attempt.weak.bbox.as_xyxy() if attempt.weak else None,
            "strong_bbox_2d": attempt.strong.bbox.as_xyxy() if attempt.strong else None,
            "weak_iou": attempt.weak_iou,
            "strong_iou": attempt.strong_iou,
            "status": status,
            "feedback": result.error if status == "error" else attempt.feedback,
            "attempt_file_id": attempt_file_id,
        }
        if self.saved_attempts.get(identifier) != row:
            _execute(
                self.client.table("attempts").upsert(row, on_conflict="id"), "save the attempt"
            )
            self.saved_attempts[identifier] = row

    def finish(self, status: str) -> None:
        _execute(
            self.client.table("runs")
            .update({"status": status, "finished_at": datetime.now(UTC).isoformat()})
            .eq("id", self.run_id),
            "finish the run",
        )

    def iter_outputs(self, run_id: str) -> Iterator[OutputRecord]:
        # Filter through attempts, keeping run_id out of the outputs table.
        offset = 0
        while True:
            rows = _execute(
                self.client.table("outputs")
                .select(f"{OUTPUT_FIELDS},attempts!inner(run_id)")
                .eq("attempts.run_id", run_id)
                .order("attempt_id")
                .range(offset, offset + PAGE_SIZE - 1),
                "read outputs",
            ).data
            for row in rows:
                row.pop("attempts", None)
                yield OutputRecord.model_validate(row)
            if len(rows) < PAGE_SIZE:
                return
            offset += len(rows)
