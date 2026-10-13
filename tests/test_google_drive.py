import hashlib
import json
import re
from urllib.parse import parse_qs, urlparse

import httplib2
import pytest
from googleapiclient.discovery import build

from google_drive import FOLDER_MIME, DriveError, GoogleDrive, create_drive_client


class MemoryDrive:
    # Run the real Drive SDK against a fake HTTP server, including resumable uploads.

    def __init__(self):
        self.objects = {
            "root-folder": {
                "id": "root-folder",
                "name": "Autodata",
                "mimeType": FOLDER_MIME,
                "parents": [],
                "trashed": False,
                "capabilities": {"canAddChildren": True},
            }
        }
        self.contents = {}
        self.sessions = {}
        self.requests = []
        self.fail_upload = False
        self.file_errors = {}
        self.list_page_size = 1000
        self.next_id = 1
        self.service = build("drive", "v3", http=self, static_discovery=True, cache_discovery=False)
        self.drive = GoogleDrive(self.service, "root-folder")

    def put_original(self, file_id, content, name="sample 1.png"):
        self.objects[file_id] = {
            "id": file_id,
            "name": name,
            "mimeType": "image/png",
            "parents": ["root-folder"],
            "trashed": False,
            "md5Checksum": hashlib.md5(content, usedforsecurity=False).hexdigest(),
            "size": str(len(content)),
        }
        self.contents[file_id] = content

    def response(self, body, status=200, **headers):
        if not isinstance(body, bytes):
            body = json.dumps(body).encode()
        return httplib2.Response({"status": status, **headers}), body

    def new_id(self, kind):
        identifier = f"{kind}-{self.next_id}"
        self.next_id += 1
        return identifier

    def request(self, uri, method="GET", body=None, headers=None, **kwargs):
        parsed = urlparse(uri)
        params = parse_qs(parsed.query)
        self.requests.append((method, uri))
        if parsed.hostname == "upload.example":
            assert method == "PUT"
            file_id = parsed.path.rsplit("/", 1)[-1]
            metadata = self.sessions[file_id]
            content = body.read() if hasattr(body, "read") else body
            self.objects[file_id] = {
                **metadata,
                "id": file_id,
                "trashed": False,
                "md5Checksum": hashlib.md5(content, usedforsecurity=False).hexdigest(),
                "size": str(len(content)),
            }
            self.contents[file_id] = content
            return self.response({"id": file_id})
        if parsed.path.startswith("/upload/"):
            if self.fail_upload:
                return self.response(
                    {"error": {"code": 403, "message": "secret server body"}}, status=403
                )
            if method == "PATCH":
                file_id = parsed.path.rsplit("/", 1)[-1]
                metadata = self.objects[file_id]
            else:
                assert method == "POST"
                file_id = self.new_id("file")
                metadata = json.loads(body)
            self.sessions[file_id] = metadata
            return self.response({}, location=f"https://upload.example/session/{file_id}")
        if parsed.path == "/drive/v3/files" and method == "GET":
            query = params["q"][0]
            parent = re.search(r"'([^']+)' in parents", query)[1]
            match = re.search(r"name = '(.*)' and trashed", query)
            name = match[1].replace("\\'", "'").replace("\\\\", "\\") if match else None
            source_match = re.search(
                r"properties has \{ key='autodata_source' and value='(.*)' \}", query
            )
            source_key = source_match[1] if source_match else None
            files = [
                value
                for value in self.objects.values()
                if parent in value.get("parents", [])
                and (name is None or value["name"] == name)
                and (
                    source_key is None
                    or value.get("properties", {}).get("autodata_source") == source_key
                )
                and not value.get("trashed")
            ]
            offset = int(params.get("pageToken", ["0"])[0])
            limit = min(int(params.get("pageSize", ["100"])[0]), self.list_page_size)
            response = {"files": files[offset : offset + limit]}
            if offset + limit < len(files):
                response["nextPageToken"] = str(offset + limit)
            return self.response(response)
        if parsed.path == "/drive/v3/files" and method == "POST":
            metadata = json.loads(body)
            file_id = self.new_id("folder")
            self.objects[file_id] = {**metadata, "id": file_id, "trashed": False}
            return self.response({"id": file_id})
        file_id = parsed.path.rsplit("/", 1)[-1]
        if file_id not in self.objects or file_id in self.file_errors:
            return self.response(
                {"error": {"message": "private server body"}},
                status=self.file_errors.get(file_id, 404),
            )
        if method == "PATCH":
            self.objects[file_id].update(json.loads(body))
            return self.response(self.objects[file_id])
        assert method == "GET", (method, uri)
        if params.get("alt") == ["media"]:
            content = self.contents[file_id]
            return self.response(content, **{"content-length": str(len(content))})
        return self.response(self.objects[file_id])


def test_drive_check_is_read_only():
    memory = MemoryDrive()
    assert memory.drive.check_connection()["folder_name"] == "Autodata"
    assert len(memory.objects) == 1
    assert all(method == "GET" for method, _ in memory.requests)


def test_original_is_uploaded_once_and_round_trips_exact_bytes(image_bytes):
    memory = MemoryDrive()
    path = "samples/image-hash.png"
    first = memory.drive.upload(path, image_bytes, "image/png", immutable=True)
    second = memory.drive.upload(path, image_bytes, "image/png", immutable=True)
    another_client = GoogleDrive(
        build("drive", "v3", http=memory, static_discovery=True, cache_discovery=False),
        "root-folder",
    )
    assert another_client.upload(path, image_bytes, "image/png", immutable=True) == first
    assert first == second
    assert memory.drive.download(first) == image_bytes
    assert sum(method == "PUT" for method, _ in memory.requests) == 1
    assert len([v for v in memory.objects.values() if v.get("mimeType") == FOLDER_MIME]) == 2


def test_attempt_updates_keep_one_file_id():
    memory = MemoryDrive()
    path = "attempts/run-1/sample 1 attempt 1.png"
    first = memory.drive.upload(path, b"first", "image/png")
    second = memory.drive.upload(path, b"second", "image/png")
    assert first == second
    assert memory.drive.download(second) == b"second"
    assert sum(method == "PUT" for method, _ in memory.requests) == 2


def test_sample_numbers_survive_new_clients_and_repeated_imports():
    memory = MemoryDrive()
    first = memory.drive.upload_sample(b"first image", "image/jpeg", source_key="coco:train2014:1")
    second = memory.drive.upload_sample(
        b"second image", "image/jpeg", source_key="coco:train2014:2"
    )
    another_client = GoogleDrive(memory.service, "root-folder")
    assert (
        another_client.upload_sample(b"second image", "image/jpeg", source_key="coco:train2014:2")
        == second
    )
    third = another_client.upload_sample(
        b"third image", "image/jpeg", source_key="coco:train2014:3"
    )
    assert (
        another_client.upload_sample(b"first image", "image/jpeg", source_key="coco:train2014:1")
        == first
    )
    assert [memory.objects[file_id]["name"] for file_id in (first, second, third)] == [
        "sample 1.jpg",
        "sample 2.jpg",
        "sample 3.jpg",
    ]
    assert another_client.sample_name(second) == "sample 2"
    assert sum(method == "PUT" for method, _ in memory.requests) == 3


def test_numbering_reads_all_pages_and_keeps_legacy_file_ids():
    memory = MemoryDrive()
    first = memory.drive.upload("samples/sample 1.jpg", b"first", "image/jpeg", immutable=True)
    last = memory.drive.upload("samples/sample 9.jpg", b"last", "image/jpeg", immutable=True)
    legacy = memory.drive.upload("samples/old-hash.jpg", b"legacy", "image/jpeg", immutable=True)
    memory.list_page_size = 1
    assert memory.drive.sample_name(legacy) == "sample 10"
    assert (
        memory.drive.upload_sample(b"legacy", "image/jpeg", source_key="coco:train2014:10")
        == legacy
    )
    new = memory.drive.upload_sample(b"new", "image/jpeg", source_key="coco:train2014:11")
    assert memory.objects[new]["name"] == "sample 11.jpg"
    assert memory.contents[legacy] == b"legacy"
    assert memory.objects[first]["name"] == "sample 1.jpg"
    assert memory.objects[last]["name"] == "sample 9.jpg"


def test_sample_upload_failure_does_not_consume_a_number():
    memory = MemoryDrive()
    memory.fail_upload = True
    with pytest.raises(DriveError):
        memory.drive.upload_sample(b"first", "image/jpeg", source_key="coco:train2014:1")
    memory.fail_upload = False
    file_id = memory.drive.upload_sample(b"first", "image/jpeg", source_key="coco:train2014:1")
    assert memory.objects[file_id]["name"] == "sample 1.jpg"


def test_original_with_wrong_content_is_never_overwritten():
    memory = MemoryDrive()
    file_id = memory.drive.upload("samples/hash.jpg", b"original", "image/jpeg", immutable=True)
    with pytest.raises(DriveError, match="differs"):
        memory.drive.upload("samples/hash.jpg", b"changed", "image/jpeg", immutable=True)
    assert memory.contents[file_id] == b"original"


@pytest.mark.parametrize("path", ["../outside.png", "/outside.png", r"samples\bad.png"])
def test_bad_paths_are_rejected_without_requests(path):
    memory = MemoryDrive()
    with pytest.raises(DriveError, match="relative"):
        memory.drive.upload(path, b"data", "image/png")
    assert memory.requests == []


def test_upload_errors_do_not_expose_google_server_bodies():
    memory = MemoryDrive()
    memory.fail_upload = True
    with pytest.raises(DriveError) as error:
        memory.drive.upload("attempts/run/attempt.png", b"data", "image/png")
    assert "HTTP 403" in str(error.value)
    assert "secret server body" not in str(error.value)


def test_missing_google_token_is_actionable_and_creates_no_folders(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GOOGLE_DRIVE_FOLDER_ID", "root-folder")
    monkeypatch.setenv("GOOGLE_DRIVE_TOKEN_FILE", str(tmp_path / "credentials/token.json"))
    with pytest.raises(DriveError, match="drive-auth"):
        create_drive_client()
    assert list(tmp_path.iterdir()) == []


def token_file(tmp_path, monkeypatch, **changes):
    path = tmp_path / "token.json"
    payload = {
        "client_id": "test-client",
        "client_secret": "test-secret",
        "refresh_token": "test-refresh",
        "token": "test-access",
        "expiry": "2099-01-01T00:00:00Z",
        "scopes": ["https://www.googleapis.com/auth/drive"],
        **changes,
    }
    path.write_text(json.dumps(payload))
    monkeypatch.setenv("GOOGLE_DRIVE_TOKEN_FILE", str(path))
    monkeypatch.setenv("GOOGLE_DRIVE_FOLDER_ID", "root-folder")
    return path


def test_cached_google_login_never_opens_a_browser(tmp_path, monkeypatch):
    memory = MemoryDrive()
    token_file(tmp_path, monkeypatch)
    monkeypatch.setattr("google_drive.build", lambda *args, **kwargs: memory.service)
    monkeypatch.setattr(
        "google_drive.InstalledAppFlow.from_client_secrets_file",
        lambda *args: pytest.fail("Normal commands must not open a browser"),
    )
    assert create_drive_client().check_connection()["folder_id"] == "root-folder"


def test_insufficient_cached_scopes_require_explicit_login(tmp_path, monkeypatch):
    token_file(tmp_path, monkeypatch, scopes=["https://www.googleapis.com/auth/drive.file"])
    with pytest.raises(DriveError, match="permissions changed"):
        create_drive_client()


def test_expired_access_token_is_refreshed_and_cached(tmp_path, monkeypatch):
    from datetime import datetime, timedelta

    memory = MemoryDrive()
    path = token_file(tmp_path, monkeypatch, expiry="2020-01-01T00:00:00Z")
    refreshed = []

    def refresh(credentials, request):
        refreshed.append(True)
        credentials.token = "refreshed-test-access"
        credentials.expiry = datetime.now() + timedelta(hours=1)

    monkeypatch.setattr("google_drive.Credentials.refresh", refresh)
    monkeypatch.setattr("google_drive.build", lambda *args, **kwargs: memory.service)
    create_drive_client()
    assert refreshed == [True]
    assert json.loads(path.read_text())["token"] == "refreshed-test-access"


def test_failed_oauth_login_does_not_print_provider_details(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from oauthlib.oauth2 import OAuth2Error

    from google_drive import authenticate_drive

    client_file = tmp_path / "client.json"
    client_file.write_text("{}")
    monkeypatch.setenv("GOOGLE_DRIVE_CLIENT_FILE", str(client_file))

    def fail(**kwargs):
        raise OAuth2Error(description="secret provider detail")

    monkeypatch.setattr(
        "google_drive.InstalledAppFlow.from_client_secrets_file",
        lambda *args: SimpleNamespace(run_local_server=fail),
    )
    with pytest.raises(DriveError) as error:
        authenticate_drive()
    assert "secret provider detail" not in str(error.value)
