import hashlib
import json
import re
from urllib.parse import parse_qs, urlparse

import httplib2
import pytest
from googleapiclient.discovery import build

from google_drive import (
    ACCEPTED_FOLDER,
    FOLDER_MIME,
    IMAGES_FOLDER,
    PREVIEWS_FOLDER,
    DriveError,
    GoogleDrive,
    create_drive_client,
)


class MemoryDrive:
    # Run the real Drive SDK against a fake HTTP server, including resumable uploads.

    def __init__(self, folders=True):
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
        self.folder_creates = []
        self.fail_upload = False
        self.fail_transport = 0
        self.list_error = None
        self.list_page_size = 1000
        self.next_id = 1
        self.service = build("drive", "v3", http=self, static_discovery=True, cache_discovery=False)
        self.drive = GoogleDrive(self.service, "root-folder")
        if folders:
            for name in (IMAGES_FOLDER, PREVIEWS_FOLDER, ACCEPTED_FOLDER):
                self.put_folder(name)

    def put_folder(self, name):
        folder_id = self.new_id("folder")
        self.objects[folder_id] = {
            "id": folder_id,
            "name": name,
            "mimeType": FOLDER_MIME,
            "parents": ["root-folder"],
            "trashed": False,
        }
        return folder_id

    def put_original(self, file_id, content, name="20.png"):
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
        if self.fail_transport:
            self.fail_transport -= 1
            raise OSError("connection reset by peer")
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
            if self.list_error:
                return self.response(
                    {"error": {"message": "private server body"}}, status=self.list_error
                )
            query = params["q"][0]
            parent = re.search(r"'([^']+)' in parents", query)[1]
            match = re.search(r"name = '(.*)' and trashed", query)
            name = match[1].replace("\\'", "'").replace("\\\\", "\\") if match else None
            files = [
                value
                for value in self.objects.values()
                if parent in value.get("parents", [])
                and (name is None or value["name"] == name)
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
            self.folder_creates.append(metadata)
            if metadata.get("mimeType") != FOLDER_MIME:
                return self.response({"error": {"message": "unexpected create"}}, status=500)
            folder_id = self.new_id("folder")
            self.objects[folder_id] = {**metadata, "id": folder_id, "trashed": False}
            return self.response({"id": folder_id})
        file_id = parsed.path.rsplit("/", 1)[-1]
        if file_id not in self.objects:
            return self.response(
                {"error": {"message": "private server body"}},
                status=404,
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
    assert len(memory.objects) == 4
    assert all(method == "GET" for method, _ in memory.requests)


def test_check_returns_both_folder_ids():
    memory = MemoryDrive()
    ids = {value["name"]: key for key, value in memory.objects.items() if key != "root-folder"}
    assert memory.drive.check_connection()["folders"] == {
        "images": ids["images"],
        "previews": ids["previews"],
        "accepted": ids["accepted"],
    }


def test_missing_folder_is_reported_and_never_created():
    memory = MemoryDrive(folders=False)
    with pytest.raises(DriveError, match="'images'"):
        memory.drive.check_connection()
    assert memory.folder_creates == []
    assert all(method == "GET" for method, _ in memory.requests)
    assert len(memory.objects) == 1


def test_original_is_uploaded_once_and_round_trips_exact_bytes(image_bytes):
    memory = MemoryDrive()
    name = "20.png"
    first = memory.drive.upload_image(name, image_bytes, "image/png")
    second = memory.drive.upload_image(name, image_bytes, "image/png")
    another_client = GoogleDrive(
        build("drive", "v3", http=memory, static_discovery=True, cache_discovery=False),
        "root-folder",
    )
    assert another_client.upload_image(name, image_bytes, "image/png") == first
    assert another_client.find_image(name) == first
    assert first == second
    assert memory.drive.download(first) == image_bytes
    assert sum(method == "PUT" for method, _ in memory.requests) == 1
    assert len([v for v in memory.objects.values() if v.get("mimeType") == FOLDER_MIME]) == 4


def test_round_updates_keep_one_file_id():
    memory = MemoryDrive()
    name = "run-1 ref 0 round 1.png"
    first = memory.drive.upload(memory.drive._folder(PREVIEWS_FOLDER), name, b"first", "image/png")
    second = memory.drive.upload(
        memory.drive._folder(PREVIEWS_FOLDER), name, b"second", "image/png"
    )
    assert first == second
    assert memory.drive.download(second) == b"second"
    assert sum(method == "PUT" for method, _ in memory.requests) == 2


def test_find_image_returns_none_for_unknown_name_and_file_id_for_known_one():
    memory = MemoryDrive()
    assert memory.drive.find_image("20.jpg") is None
    file_id = memory.drive.upload_image("20.jpg", b"jpeg", "image/jpeg")
    assert memory.drive.find_image("20.jpg") == file_id
    assert memory.drive.find_image("21.jpg") is None


def test_original_with_wrong_content_is_never_overwritten():
    memory = MemoryDrive()
    name = "20.jpg"
    file_id = memory.drive.upload_image(name, b"original", "image/jpeg")
    with pytest.raises(DriveError, match="differs"):
        memory.drive.upload_image(name, b"changed", "image/jpeg")
    assert memory.contents[file_id] == b"original"


@pytest.mark.parametrize("name", ["", "../outside.png", "/outside.png", r"images\bad.png"])
def test_bad_names_are_rejected_without_requests(name):
    memory = MemoryDrive()
    with pytest.raises(DriveError, match="names"):
        memory.drive.upload(IMAGES_FOLDER, name, b"data", "image/png")
    assert memory.requests == []


def test_upload_errors_do_not_expose_google_server_bodies():
    memory = MemoryDrive()
    memory.fail_upload = True
    with pytest.raises(DriveError) as error:
        memory.drive.upload(
            memory.drive._folder(PREVIEWS_FOLDER), "run round.png", b"data", "image/png"
        )
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


def test_run_folder_is_created_once_inside_previews_and_reused_on_resume():
    memory = MemoryDrive()
    first = memory.drive.run_folder(PREVIEWS_FOLDER, "run-1")
    again = GoogleDrive(memory.service, "root-folder").run_folder(PREVIEWS_FOLDER, "run-1")
    assert first == again
    assert len(memory.folder_creates) == 1
    created = memory.objects[first]
    assert created["name"] == "run-1" and created["mimeType"] == FOLDER_MIME
    assert memory.objects[created["parents"][0]]["name"] == PREVIEWS_FOLDER


def test_dropped_connection_is_retried_then_the_upload_succeeds(drive_waits):
    memory = MemoryDrive()
    memory.fail_transport = 1
    file_id = memory.drive.upload_image("20.png", b"png-bytes", "image/png")
    assert memory.contents[file_id] == b"png-bytes"
    assert drive_waits == [2]


def test_persistent_dropped_connection_reports_attempts_without_server_details(drive_waits):
    memory = MemoryDrive()
    memory.fail_transport = 10
    with pytest.raises(DriveError, match="OSError after 3 attempts"):
        memory.drive.upload_image("20.png", b"png-bytes", "image/png")
    assert drive_waits == [2, 5]


def test_download_retries_a_dropped_connection(drive_waits):
    memory = MemoryDrive()
    memory.put_original("file-1", b"original")
    memory.fail_transport = 1
    assert memory.drive.download("file-1") == b"original"
    assert drive_waits == [2]
