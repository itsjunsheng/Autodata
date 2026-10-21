# Authenticate with Google; store originals in images/ and round previews in previews/ in Drive.
# The user creates images/, previews/ and accepted/ by hand; the code creates only one folder
# per run inside previews/ and accepted/.

import hashlib
import os
import re
from io import BytesIO
from pathlib import Path
from time import sleep

import httplib2
from google.auth.exceptions import GoogleAuthError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_httplib2 import AuthorizedHttp
from google_auth_oauthlib.flow import InstalledAppFlow, WSGITimeoutError
from googleapiclient.discovery import Resource, build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload
from oauthlib.oauth2 import OAuth2Error
from requests.exceptions import RequestException

SCOPES = ["https://www.googleapis.com/auth/drive"]
FOLDER_MIME = "application/vnd.google-apps.folder"
IMAGES_FOLDER = "images"
PREVIEWS_FOLDER = "previews"
ACCEPTED_FOLDER = "accepted"
IMAGE_SUFFIXES = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}
RETRY_DELAYS = (2, 5)  # seconds before the second and third attempt of a dropped request


class DriveError(RuntimeError):
    # A Drive failure with a message safe to print.
    pass


def _retry_transport(action, operation: str):
    # Retry a call that failed below HTTP: a dropped connection, a timeout, or a token refresh.
    for delay in (*RETRY_DELAYS, None):
        try:
            return action()
        except (GoogleAuthError, OSError, httplib2.HttpLib2Error) as error:
            if delay is None:
                raise DriveError(
                    f"Google Drive {operation} failed ({type(error).__name__} after "
                    f"{len(RETRY_DELAYS) + 1} attempts). Check authentication and connectivity."
                ) from error
            sleep(delay)


def validate_file_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise DriveError("Expected a Google Drive file/folder ID, not a sharing URL.")
    return value


def _token_path() -> Path:
    return Path(os.getenv("GOOGLE_DRIVE_TOKEN_FILE", "credentials/google_token.json"))


def _save_token(credentials: Credentials) -> None:
    path = _token_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(credentials.to_json(), encoding="utf-8")
    temporary.replace(path)


def authenticate_drive() -> None:
    # Explicit, one-time browser login; regular commands never open a browser.
    client_file = Path(os.getenv("GOOGLE_DRIVE_CLIENT_FILE", "credentials/google_client.json"))
    if not client_file.is_file():
        raise DriveError(
            "Download a Google OAuth Desktop app client JSON to "
            "GOOGLE_DRIVE_CLIENT_FILE. See README.md."
        )
    try:
        flow = InstalledAppFlow.from_client_secrets_file(str(client_file), SCOPES)
        credentials = flow.run_local_server(port=0, timeout_seconds=180, prompt="consent")
        _save_token(credentials)
    except (
        GoogleAuthError,
        OAuth2Error,
        RequestException,
        WSGITimeoutError,
        OSError,
        ValueError,
    ) as error:
        raise DriveError(
            "Google login failed. Check your OAuth Desktop app configuration."
        ) from error


def create_drive_client() -> "GoogleDrive":
    folder_id = validate_file_id(os.getenv("GOOGLE_DRIVE_FOLDER_ID", "").strip())
    path = _token_path()
    if not path.is_file():
        raise DriveError("Google is not authenticated. Run: python src/main.py drive-auth")
    try:
        credentials = Credentials.from_authorized_user_file(str(path))
        if not credentials.has_scopes(SCOPES):
            raise DriveError("Google permissions changed. Run: python src/main.py drive-auth")
        if not credentials.valid:
            if not credentials.refresh_token:
                raise DriveError("Google login expired. Run: python src/main.py drive-auth")
            credentials.refresh(Request())
            _save_token(credentials)
        service = build(
            "drive",
            "v3",
            http=AuthorizedHttp(credentials, http=httplib2.Http(timeout=60)),
            cache_discovery=False,
            static_discovery=True,
        )
    except (GoogleAuthError, OSError, ValueError, httplib2.HttpLib2Error) as error:
        raise DriveError(
            "Google authentication failed. Run: python src/main.py drive-auth"
        ) from error
    return GoogleDrive(service, folder_id)


class GoogleDrive:
    # Use only two hand-made folders inside the configured root, reusing files by name.

    def __init__(self, service: Resource, folder_id: str):
        self.files = service.files()
        self.folder_id = validate_file_id(folder_id)
        self.folders: dict[str, str] = {}

    def _execute(self, request):
        try:
            return _retry_transport(lambda: request.execute(num_retries=2), "request")
        except HttpError as error:
            raise DriveError(
                f"Google Drive request failed (HTTP {error.resp.status}). "
                "Check account permissions, available storage, and API quotas."
            ) from error

    def check_connection(self) -> dict:
        # Check the root and its two folders without uploading files or creating folders.
        folder = self._execute(
            self.files.get(
                fileId=self.folder_id,
                fields="id,name,mimeType,trashed,capabilities(canAddChildren)",
                supportsAllDrives=True,
            )
        )
        if folder.get("trashed") or folder.get("mimeType") != FOLDER_MIME:
            raise DriveError("GOOGLE_DRIVE_FOLDER_ID must identify an existing, untrashed folder.")
        if not folder.get("capabilities", {}).get("canAddChildren"):
            raise DriveError("Your Google account cannot upload into the configured folder.")
        return {
            "connected": True,
            "folder_id": self.folder_id,
            "folder_name": folder["name"],
            "folders": {
                name: self._folder(name)
                for name in (IMAGES_FOLDER, PREVIEWS_FOLDER, ACCEPTED_FOLDER)
            },
        }

    def _find(self, parent_id: str, name: str) -> dict | None:
        escaped = name.replace("\\", "\\\\").replace("'", "\\'")
        response = self._execute(
            self.files.list(
                q=f"'{parent_id}' in parents and name = '{escaped}' and trashed = false",
                fields="files(id,mimeType,md5Checksum,size),nextPageToken,incompleteSearch",
                pageSize=2,
                spaces="drive",
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
            )
        )
        if response.get("incompleteSearch"):
            raise DriveError("Drive returned an incomplete file search. Try again.")
        matches = response.get("files", [])
        if len(matches) > 1 or response.get("nextPageToken"):
            raise DriveError(f"Duplicate files in the Drive folder for {name}. Resolve them first.")
        return matches[0] if matches else None

    def find_image(self, name: str) -> str | None:
        # Find a completed upload by its COCO file name even if its Supabase insert failed.
        found = self._find(self._folder(IMAGES_FOLDER), name)
        if found is None:
            return None
        if found.get("mimeType") not in IMAGE_SUFFIXES or int(found.get("size", 0)) <= 0:
            raise DriveError("The stored file for this image name is not a non-empty image.")
        return validate_file_id(found["id"])

    def _folder(self, name: str) -> str:
        # Look up a folder the user made by hand.
        if name not in self.folders:
            found = self._find(self.folder_id, name)
            if found is None or found.get("mimeType") != FOLDER_MIME:
                raise DriveError(
                    f"Create a folder named '{name}' inside the configured Drive folder."
                )
            self.folders[name] = validate_file_id(found["id"])
        return self.folders[name]

    def run_folder(self, parent: str, run_id: str) -> str:
        # The one kind of folder the code creates: <parent>/<run id>, reused when a run resumes.
        key = f"{parent}/{run_id}"
        if key not in self.folders:
            parent_id = self._folder(parent)
            found = self._find(parent_id, run_id)
            if found is None:
                found = self._execute(
                    self.files.create(
                        body={"name": run_id, "parents": [parent_id], "mimeType": FOLDER_MIME},
                        fields="id",
                        supportsAllDrives=True,
                    )
                )
            elif found.get("mimeType") != FOLDER_MIME:
                raise DriveError("A file occupies the run's folder name.")
            self.folders[key] = validate_file_id(found["id"])
        return self.folders[key]

    def upload(
        self, parent: str, name: str, content: bytes, content_type: str, *, immutable: bool = False
    ) -> str:
        # Reuse a matching original; update an existing round preview without making copies.
        if not name or "/" in name or "\\" in name:
            raise DriveError("Drive file names must not be empty or contain slashes.")
        found = self._find(parent, name)
        if found and found.get("mimeType") == FOLDER_MIME:
            raise DriveError("A folder occupies the expected Drive file name.")
        existing_id = found["id"] if found else None
        if immutable and existing_id:
            metadata = found
            checksum = hashlib.md5(content, usedforsecurity=False).hexdigest()
            if metadata.get("md5Checksum") != checksum or int(metadata.get("size", -1)) != len(
                content
            ):
                raise DriveError(
                    "An existing original differs from its content hash. Do not overwrite it."
                )
            return validate_file_id(existing_id)
        media = MediaIoBaseUpload(BytesIO(content), mimetype=content_type, resumable=True)
        if existing_id:
            request = self.files.update(
                fileId=existing_id, media_body=media, fields="id", supportsAllDrives=True
            )
        else:
            request = self.files.create(
                body={"name": name, "parents": [parent], "mimeType": content_type},
                media_body=media,
                fields="id",
                supportsAllDrives=True,
            )
        return validate_file_id(self._execute(request)["id"])

    def upload_image(self, name: str, content: bytes, content_type: str) -> str:
        # Store an original under its real COCO file name; a repeat upload reuses the file.
        if content_type not in IMAGE_SUFFIXES:
            raise DriveError("Images must be JPEG, PNG, or WebP.")
        return self.upload(self._folder(IMAGES_FOLDER), name, content, content_type, immutable=True)

    def download(self, file_id: str) -> bytes:
        # Read a file into memory, with bounded retries and no disk cache.
        request = self.files.get_media(fileId=validate_file_id(file_id), supportsAllDrives=True)

        def fetch() -> bytes:
            buffer = BytesIO()
            downloader = MediaIoBaseDownload(buffer, request)
            done = False
            while not done:
                _, done = downloader.next_chunk(num_retries=2)
            return buffer.getvalue()

        try:
            return _retry_transport(fetch, "download")
        except HttpError as error:
            raise DriveError(f"Google Drive download failed (HTTP {error.resp.status}).") from error
