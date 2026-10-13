# Authenticate with Google and store sample images and attempt images in Drive.

import hashlib
import os
import re
from io import BytesIO
from pathlib import Path, PurePosixPath

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
IMAGE_SUFFIXES = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}
SAMPLE_NAME = re.compile(r"sample ([1-9]\d*)\.(jpg|jpeg|png|webp)")


class DriveError(RuntimeError):
    # A Drive failure with a message safe to print.
    pass


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
    # Use only the configured root folder, reusing files by their relative paths.

    def __init__(self, service: Resource, folder_id: str):
        self.files = service.files()
        self.folder_id = validate_file_id(folder_id)
        self.folders: dict[str, str] = {"": self.folder_id}
        self.sample_files: dict[tuple[str, int], dict] | None = None
        self.next_sample_number = 1

    def _execute(self, request, *, allow_missing: bool = False):
        try:
            return request.execute(num_retries=2)
        except HttpError as error:
            if allow_missing and error.resp.status == 404:
                return None
            raise DriveError(
                f"Google Drive request failed (HTTP {error.resp.status}). "
                "Check account permissions, available storage, and API quotas."
            ) from error
        except (GoogleAuthError, OSError, httplib2.HttpLib2Error) as error:
            raise DriveError(
                "Google Drive request failed. Check authentication and connectivity."
            ) from error

    def check_connection(self) -> dict:
        # Check the root folder without uploading files or creating folders.
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
        return {"connected": True, "folder_id": self.folder_id, "folder_name": folder["name"]}

    def _find(
        self, parent_id: str, name: str | None = None, *, source_key: str | None = None
    ) -> dict | None:
        value = name if name is not None else source_key
        if value is None:
            raise ValueError("A file name or source key is required.")
        escaped = value.replace("\\", "\\\\").replace("'", "\\'")
        condition = (
            f"name = '{escaped}'"
            if name is not None
            else f"properties has {{ key='autodata_source' and value='{escaped}' }}"
        )
        response = self._execute(
            self.files.list(
                q=f"'{parent_id}' in parents and {condition} and trashed = false",
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
            raise DriveError(
                f"Duplicate files in the Drive folder for {value}. Resolve them first."
            )
        return matches[0] if matches else None

    def image_exists(self, file_id: str) -> bool:
        # Read file metadata only; permission and network errors still stop the import.
        metadata = self._execute(
            self.files.get(
                fileId=validate_file_id(file_id),
                fields="mimeType,trashed,size",
                supportsAllDrives=True,
            ),
            allow_missing=True,
        )
        if metadata is None or metadata.get("trashed"):
            return False
        if metadata.get("mimeType") not in IMAGE_SUFFIXES or int(metadata.get("size", 0)) <= 0:
            raise DriveError("The linked sample file is not a non-empty image.")
        return True

    def find_sample(self, source_key: str) -> str | None:
        # Find a completed image upload even if its Supabase insert failed.
        found = self._find(self._folder(("samples",)), source_key=source_key)
        if found is None:
            return None
        if found.get("mimeType") not in IMAGE_SUFFIXES or int(found.get("size", 0)) <= 0:
            raise DriveError("The stored source identifier points to an invalid sample image.")
        return validate_file_id(found["id"])

    def _folder(self, parts: tuple[str, ...]) -> str:
        parent = self.folder_id
        for index, name in enumerate(parts):
            key = "/".join(parts[: index + 1])
            if key not in self.folders:
                found = self._find(parent, name)
                if found and found["mimeType"] != FOLDER_MIME:
                    raise DriveError(f"A file occupies the expected Drive folder: {key}")
                if not found:
                    found = self._execute(
                        self.files.create(
                            body={"name": name, "mimeType": FOLDER_MIME, "parents": [parent]},
                            fields="id",
                            supportsAllDrives=True,
                        )
                    )
                self.folders[key] = validate_file_id(found["id"])
            parent = self.folders[key]
        return parent

    def upload(
        self, path: str, content: bytes, content_type: str, *, immutable: bool = False
    ) -> str:
        # Reuse clean sample images; update an existing attempt image without making copies.
        relative = PurePosixPath(path)
        if relative.is_absolute() or ".." in relative.parts or not relative.name or "\\" in path:
            raise DriveError("Drive upload paths must be relative to the configured root folder.")
        parent = self._folder(relative.parts[:-1])
        found = self._find(parent, relative.name)
        if found and found.get("mimeType") == FOLDER_MIME:
            raise DriveError("A folder occupies the expected Drive file path.")
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
                body={"name": relative.name, "parents": [parent], "mimeType": content_type},
                media_body=media,
                fields="id",
                supportsAllDrives=True,
            )
        file_id = validate_file_id(self._execute(request)["id"])
        return file_id

    def _load_sample_files(self) -> None:
        # Read lightweight metadata once per import, including every Drive page.
        if self.sample_files is not None:
            return
        parent = self._folder(("samples",))
        files = {}
        next_number = 1
        page_token = None
        while True:
            response = self._execute(
                self.files.list(
                    q=f"'{parent}' in parents and trashed = false",
                    fields="files(id,name,mimeType,md5Checksum,size),nextPageToken,incompleteSearch",
                    pageSize=1000,
                    pageToken=page_token,
                    spaces="drive",
                    supportsAllDrives=True,
                    includeItemsFromAllDrives=True,
                )
            )
            if response.get("incompleteSearch"):
                raise DriveError("Drive returned an incomplete sample listing. Try again.")
            for metadata in response.get("files", []):
                match = SAMPLE_NAME.fullmatch(metadata["name"])
                if match:
                    next_number = max(next_number, int(match[1]) + 1)
                if metadata.get("mimeType") in IMAGE_SUFFIXES and metadata.get("md5Checksum"):
                    key = (metadata["md5Checksum"], int(metadata["size"]))
                    files.setdefault(key, metadata)
            page_token = response.get("nextPageToken")
            if not page_token:
                break
        self.sample_files = files
        self.next_sample_number = next_number

    def _number_sample(self, metadata: dict) -> str:
        # Rename a legacy image without changing its file ID or image bytes.
        if not SAMPLE_NAME.fullmatch(metadata["name"]):
            name = f"sample {self.next_sample_number}{IMAGE_SUFFIXES[metadata['mimeType']]}"
            if self._find(self._folder(("samples",)), name):
                raise DriveError("Sample numbering changed. Use one importer at a time.")
            self._execute(
                self.files.update(
                    fileId=metadata["id"], body={"name": name}, fields="id", supportsAllDrives=True
                )
            )
            metadata["name"] = name
            self.next_sample_number += 1
        return PurePosixPath(metadata["name"]).stem

    def upload_sample(self, content: bytes, content_type: str, *, source_key: str) -> str:
        # Name new images sequentially and reuse identical images across imports.
        if content_type not in IMAGE_SUFFIXES:
            raise DriveError("Sample images must be JPEG, PNG, or WebP.")
        self._load_sample_files()
        key = (hashlib.md5(content, usedforsecurity=False).hexdigest(), len(content))
        existing = self.sample_files.get(key)
        if existing:
            self._number_sample(existing)
            file_id = validate_file_id(existing["id"])
        else:
            name = f"sample {self.next_sample_number}{IMAGE_SUFFIXES[content_type]}"
            file_id = self.upload(f"samples/{name}", content, content_type, immutable=True)
            self.sample_files[key] = {"id": file_id, "name": name, "mimeType": content_type}
            self.next_sample_number += 1
        self._execute(
            self.files.update(
                fileId=file_id,
                body={"properties": {"autodata_source": source_key}},
                fields="id",
                supportsAllDrives=True,
            )
        )
        return file_id

    def sample_name(self, file_id: str) -> str:
        # Use the original's sample number when naming its attempt images.
        metadata = self._execute(
            self.files.get(
                fileId=validate_file_id(file_id),
                fields="id,name,mimeType,md5Checksum,size,trashed",
                supportsAllDrives=True,
            )
        )
        if metadata.get("trashed") or metadata.get("mimeType") not in IMAGE_SUFFIXES:
            raise DriveError("The sample file must be an existing image.")
        if not SAMPLE_NAME.fullmatch(metadata["name"]):
            self._load_sample_files()
            key = (metadata.get("md5Checksum"), int(metadata.get("size", -1)))
            existing = self.sample_files.get(key)
            if existing is None or existing["id"] != file_id:
                raise DriveError("The original image is missing from the samples folder.")
            return self._number_sample(existing)
        return PurePosixPath(metadata["name"]).stem

    def download(self, file_id: str) -> bytes:
        # Read a file into memory, with bounded retries and no disk cache.
        request = self.files.get_media(fileId=validate_file_id(file_id), supportsAllDrives=True)
        buffer = BytesIO()
        downloader = MediaIoBaseDownload(buffer, request)
        try:
            done = False
            while not done:
                _, done = downloader.next_chunk(num_retries=2)
        except HttpError as error:
            raise DriveError(f"Google Drive download failed (HTTP {error.resp.status}).") from error
        except (GoogleAuthError, OSError, httplib2.HttpLib2Error) as error:
            raise DriveError(
                "Google Drive download failed. Check authentication and connectivity."
            ) from error
        return buffer.getvalue()
