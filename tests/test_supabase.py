import json

import httpx
import pytest
from conftest import make_settings
from supabase import ClientOptions, create_client

import supabase_client
from main import main
from supabase_client import (
    SupabaseError,
    SupabaseStore,
    check_supabase_connection,
    create_supabase_client,
)

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
        self.tables = {name: {} for name in ("refcoco_annotations", "runs", "rounds")}
        self.requests = []
        self.fail_round_write = False
        self.fail_annotation_write = False
        self.timeout_writes = 0
        self.client = mock_client(self.respond)

    def seed(self, record):
        self.tables["refcoco_annotations"][record.ref_id] = record.model_dump(mode="json")

    def respond(self, request):
        self.requests.append(request)
        assert request.url.path.startswith("/rest/v1/"), "Supabase Storage must never be called"
        table = request.url.path.removeprefix("/rest/v1/")
        if request.method == "GET":
            rows = list(self.tables[table].values())
            for ordering in reversed(request.url.params.get("order", "ref_id.asc").split(",")):
                field, direction = ordering.split(".")[:2]
                rows.sort(key=lambda row: row.get(field, ""), reverse=direction == "desc")
            for field, value in request.url.params.items():
                if value.startswith("eq."):
                    rows = [row for row in rows if str(row[field]) == value[3:]]
                elif value.startswith("in.("):
                    allowed = value[4:-1].split(",")
                    rows = [row for row in rows if str(row[field]) in allowed]
            select = request.url.params.get("select", "")
            if table == "refcoco_annotations":
                fields = select.split(",")
                rows = [{field: row[field] for field in fields} for row in rows]
            elif "refcoco_annotations(" in select:
                embedded = select.partition("(")[2].rstrip(")").split(",")
                rows = [
                    {
                        "referring_expression": row["referring_expression"],
                        "refcoco_annotations": {
                            field: self.tables["refcoco_annotations"][row["ref_id"]][field]
                            for field in embedded
                        },
                    }
                    for row in rows
                ]
            offset = int(request.url.params.get("offset", 0))
            limit = int(request.url.params.get("limit", len(rows)))
            return httpx.Response(
                200,
                json=rows[offset : offset + limit],
                headers={"Content-Range": f"*/{len(rows)}"},
            )
        if request.method == "DELETE":
            wanted = {f: v[3:] for f, v in request.url.params.items() if v.startswith("eq.")}
            for key in [
                key
                for key, row in self.tables[table].items()
                if all(str(row.get(f)) == v for f, v in wanted.items())
            ]:
                del self.tables[table][key]
            return httpx.Response(200, json=[])
        row = json.loads(request.content)
        if self.timeout_writes:
            self.timeout_writes -= 1
            raise httpx.ReadTimeout("timed out", request=request)
        if (self.fail_round_write and table == "rounds") or (
            self.fail_annotation_write and table == "refcoco_annotations"
        ):
            return httpx.Response(
                403, json={"code": "42501", "message": TEST_KEY, "hint": None, "details": None}
            )
        if request.method == "PATCH":
            if table == "refcoco_annotations":
                existing = self.tables[table].get(int(request.url.params["ref_id"][3:]))
            else:
                existing = self.tables[table].get(request.url.params["id"][3:])
            if existing is None:
                return httpx.Response(200, json=[])
            existing.update(row)
            return httpx.Response(200, json=[existing])
        assert request.method == "POST"
        key = {
            "rounds": lambda: (row["run_id"], row["ref_id"], row["round_number"]),
            "refcoco_annotations": lambda: row["ref_id"],
            "runs": lambda: row["id"],
        }[table]()
        if "resolution=ignore-duplicates" in request.headers.get("prefer", ""):
            self.tables[table].setdefault(key, row)
        else:
            self.tables[table][key] = row
        return httpx.Response(200, json=[row])


def test_connection_is_read_only_and_never_requires_storage():
    memory = MemorySupabase()
    report = check_supabase_connection(memory.client)
    assert report == {"connected": True, "ready": True, "schema_issues": []}
    assert len(memory.requests) == 3
    assert all(request.method == "GET" for request in memory.requests)
    assert TEST_KEY not in json.dumps(report)


@pytest.mark.parametrize("code", ["PGRST205", "42703"])
def test_missing_tables_or_columns_are_reported(code):
    def respond(request):
        if request.url.path.endswith("/rounds"):
            return httpx.Response(
                404, json={"code": code, "message": "Missing schema", "hint": None, "details": None}
            )
        return httpx.Response(200, json=[])

    report = check_supabase_connection(mock_client(respond))
    assert report["connected"] and not report["ready"]
    assert report["schema_issues"] == ["rounds"]


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


def test_cli_missing_key_creates_no_local_files(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SUPABASE_URL", PROJECT_URL)
    monkeypatch.setenv("SUPABASE_API_KEY", "")
    assert main(["supabase-check"]) == 1
    assert "Set SUPABASE_API_KEY" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []


def test_transient_write_timeout_is_retried(monkeypatch):
    memory = MemorySupabase()
    memory.timeout_writes = 1
    delays = []
    monkeypatch.setattr(supabase_client, "sleep", delays.append)
    store = SupabaseStore(memory.client)
    store.start(make_settings(), 0, 1)
    assert memory.tables["runs"][store.run_id]["status"] == "running"
    assert delays == [2]


def test_persistent_write_timeout_names_the_failure_without_the_key(monkeypatch):
    memory = MemorySupabase()
    memory.timeout_writes = 10
    delays = []
    monkeypatch.setattr(supabase_client, "sleep", delays.append)
    with pytest.raises(SupabaseError) as error:
        SupabaseStore(memory.client).start(make_settings(), 0, 1)
    assert "ReadTimeout" in str(error.value)
    assert TEST_KEY not in str(error.value)
    assert delays == [2, 5]


def test_database_rejection_is_not_retried_and_reports_its_code(monkeypatch, sample_record):
    memory = MemorySupabase()
    memory.fail_annotation_write = True
    monkeypatch.setattr(supabase_client, "sleep", lambda _: pytest.fail("must not retry"))
    with pytest.raises(SupabaseError) as error:
        SupabaseStore(memory.client).save_annotation(sample_record)
    assert "42501" in str(error.value)
    assert TEST_KEY not in str(error.value)
