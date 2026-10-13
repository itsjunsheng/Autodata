import json

import httpx
import pytest
from supabase import ClientOptions, create_client

from main import main
from supabase_client import SupabaseError, check_supabase_connection, create_supabase_client

PROJECT_URL = "https://aqmpmnnnvpmthpfwfibb.supabase.co"
TEST_KEY = "sb_secret_test_only"


def mock_client(handler):
    return create_client(
        PROJECT_URL,
        TEST_KEY,
        options=ClientOptions(
            auto_refresh_token=False,
            persist_session=False,
            httpx_client=httpx.Client(transport=httpx.MockTransport(handler)),
        ),
    )


class MemorySupabase:
    # Exercise the real Python SDK through its HTTP interface.

    def __init__(self):
        self.tables = {name: {} for name in ("samples", "runs", "attempts", "outputs")}
        self.requests = []
        self.fail_attempt_write = False
        self.fail_sample_write = False
        self.fail_output_write = False
        self.client = mock_client(self.respond)

    def seed(self, record):
        self.tables["samples"][str(record.id)] = record.model_dump(mode="json")

    def outputs_for(self, run_id):
        return [
            row.copy()
            for row in sorted(self.tables["outputs"].values(), key=lambda row: row["attempt_id"])
            if self.tables["attempts"][row["attempt_id"]]["run_id"] == run_id
        ]

    def respond(self, request):
        self.requests.append(request)
        assert request.url.path.startswith("/rest/v1/"), "Supabase Storage must never be called"
        table = request.url.path.removeprefix("/rest/v1/")
        if request.method == "GET":
            if table == "outputs":
                run_id = request.url.params.get("attempts.run_id")
                rows = self.outputs_for(run_id[3:]) if run_id else []
                if "attempts!inner" in request.url.params.get("select", ""):
                    rows = [dict(row, attempts={"run_id": run_id[3:]}) for row in rows]
            else:
                rows = list(self.tables[table].values())
                for ordering in reversed(request.url.params.get("order", "id.asc").split(",")):
                    field, direction = ordering.split(".")[:2]
                    rows.sort(key=lambda row: row.get(field, ""), reverse=direction == "desc")
                for field in ("id", "dataset"):
                    value = request.url.params.get(field)
                    if value:
                        rows = [row for row in rows if row[field] == value[3:]]
                if table == "samples":
                    fields = request.url.params["select"].split(",")
                    rows = [{field: row[field] for field in fields} for row in rows]
            offset = int(request.url.params.get("offset", 0))
            limit = int(request.url.params.get("limit", len(rows)))
            return httpx.Response(
                200,
                json=rows[offset : offset + limit],
                headers={"Content-Range": f"*/{len(rows)}"},
            )
        row = json.loads(request.content)
        if (
            (self.fail_attempt_write and table == "attempts")
            or (self.fail_sample_write and table == "samples")
            or (self.fail_output_write and table == "attempts" and row.get("status") == "accepted")
        ):
            return httpx.Response(
                403, json={"code": "42501", "message": TEST_KEY, "hint": None, "details": None}
            )
        if request.method == "PATCH":
            existing = self.tables[table].get(request.url.params["id"][3:])
            if existing is None:
                return httpx.Response(200, json=[])
            existing.update(row)
            return httpx.Response(200, json=[existing])
        assert request.method == "POST"
        if "resolution=ignore-duplicates" in request.headers.get("prefer", ""):
            self.tables[table].setdefault(row["id"], row)
        else:
            self.tables[table][row["id"]] = row
        # Simulate the setup.sql trigger; real PostgreSQL is checked separately.
        if table == "attempts" and row["status"] == "accepted":
            sample_record = self.tables["samples"][row["sample_id"]]
            self.tables["outputs"].setdefault(
                row["id"],
                {
                    "attempt_id": row["id"],
                    "image_file_id": sample_record["image_file_id"],
                    "width": sample_record["width"],
                    "height": sample_record["height"],
                    "referring_expression": row["referring_expression"],
                    "bbox_2d": list(sample_record["bbox_2d"]),
                },
            )
        return httpx.Response(200, json=[row])


def test_connection_is_read_only_and_never_requires_storage():
    memory = MemorySupabase()
    report = check_supabase_connection(memory.client)
    assert report == {"connected": True, "ready": True, "schema_issues": []}
    assert len(memory.requests) == 4
    assert all(request.method == "GET" for request in memory.requests)
    assert TEST_KEY not in json.dumps(report)


@pytest.mark.parametrize("code", ["PGRST205", "42703"])
def test_missing_tables_or_columns_are_reported(code):
    def respond(request):
        if request.url.path.endswith("/attempts"):
            return httpx.Response(
                404, json={"code": code, "message": "Missing schema", "hint": None, "details": None}
            )
        return httpx.Response(200, json=[])

    report = check_supabase_connection(mock_client(respond))
    assert report["connected"] and not report["ready"]
    assert report["schema_issues"] == ["attempts"]


@pytest.mark.parametrize("failure", ["database", "network"])
def test_connection_errors_do_not_print_server_bodies_or_credentials(failure):
    def respond(request):
        if failure == "network":
            raise httpx.ConnectError(TEST_KEY, request=request)
        return httpx.Response(
            403, json={"code": "42501", "message": TEST_KEY, "hint": None, "details": None}
        )

    with pytest.raises(SupabaseError) as error:
        check_supabase_connection(mock_client(respond))
    assert TEST_KEY not in str(error.value)


@pytest.mark.parametrize("key", ["", "your-api-key", "sb_publishable_test"])
def test_backend_key_is_required(monkeypatch, key):
    monkeypatch.setenv("SUPABASE_URL", PROJECT_URL)
    monkeypatch.setenv("SUPABASE_API_KEY", key)
    with pytest.raises(SupabaseError, match="key"):
        create_supabase_client()


@pytest.mark.parametrize(
    "url", ["", "http://example.com", PROJECT_URL + "/dashboard", PROJECT_URL + "?apikey=secret"]
)
def test_api_url_has_no_path_or_query(monkeypatch, url):
    monkeypatch.setenv("SUPABASE_URL", url)
    monkeypatch.setenv("SUPABASE_API_KEY", TEST_KEY)
    with pytest.raises(SupabaseError, match="HTTPS API URL"):
        create_supabase_client()


@pytest.mark.parametrize("ready", [True, False])
def test_cli_reports_connection_without_calling_models(monkeypatch, capsys, ready):
    monkeypatch.setattr("main.create_supabase_client", lambda: object())
    monkeypatch.setattr(
        "main.check_supabase_connection", lambda _: {"connected": True, "ready": ready}
    )
    assert main(["supabase-check"]) == (0 if ready else 1)
    assert '"connected": true' in capsys.readouterr().out


def test_cli_missing_key_creates_no_outputs(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SUPABASE_URL", PROJECT_URL)
    monkeypatch.setenv("SUPABASE_API_KEY", "")
    assert main(["supabase-check"]) == 1
    assert "Set SUPABASE_API_KEY" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []
