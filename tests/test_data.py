import json
from io import BytesIO

import httpx
import pytest
from httpx import Client as HTTPClient
from pydantic import ValidationError
from test_google_drive import MemoryDrive
from test_supabase import MemorySupabase

from dataset import (
    DatasetError,
    import_refcoco,
    iter_references,
    load_image,
    reference_from_row,
    request_source,
)
from google_drive import DriveError, GoogleDrive
from models import Box, GroundingAnswer, from_coco_xywh, normalize_bbox
from supabase_client import SupabaseError, SupabaseStore
from verifier import iou


def test_coco_conversion_uses_width_height_not_bottom_right():
    box = from_coco_xywh([20, 20, 80, 60], 200, 100)
    assert box == Box(x_min=100, y_min=200, x_max=500, y_max=800)
    assert box.pixels(200, 100) == (20, 20, 100, 80)


def test_normalization_preserves_non_square_image_geometry_and_iou():
    pixel_box = [80, 80, 400, 320]
    normalised = normalize_bbox(pixel_box, width=800, height=400)
    assert normalised == [100, 200, 500, 800]
    original = Box.from_pixels(pixel_box, width=800, height=400)
    decoded = Box.from_xyxy(normalised)
    assert decoded.pixels(800, 400) == pytest.approx(pixel_box)
    assert iou(original, decoded) == pytest.approx(1)
    assert normalize_bbox([0, 0, 800, 400], width=800, height=400) == [0, 0, 1000, 1000]


def test_outward_rounding_preserves_tiny_ground_truth_box():
    pixel_box = [10.2, 20.3, 10.4, 20.5]
    exact = normalize_bbox(pixel_box, width=2000, height=1000)
    rounded = normalize_bbox(pixel_box, width=2000, height=1000, round_outward=True)
    assert rounded == [5, 20, 6, 21]
    assert rounded[0] <= exact[0] < exact[2] <= rounded[2]
    assert rounded[1] <= exact[1] < exact[3] <= rounded[3]


@pytest.mark.parametrize(
    "width,height", [(0, 400), (800, -1), (float("inf"), 400), (800, float("nan"))]
)
def test_normalizer_rejects_invalid_dimensions(width, height):
    with pytest.raises(ValueError, match="finite and positive"):
        normalize_bbox([80, 80, 400, 320], width, height)


@pytest.mark.parametrize(
    "coords", [[80, 80, 400], [400, 80, 80, 320], [-1, 80, 400, 320], [80, 80, 801, 320]]
)
def test_normalizer_rejects_invalid_boxes_instead_of_clipping(coords):
    with pytest.raises(ValueError):
        normalize_bbox(coords, width=800, height=400)


def test_iou_handles_overlap_and_touching_edges():
    first = Box(x_min=0, y_min=0, x_max=500, y_max=500)
    second = Box(x_min=250, y_min=250, x_max=750, y_max=750)
    touching = Box(x_min=500, y_min=500, x_max=1000, y_max=1000)
    assert iou(first, first) == 1
    assert iou(first, second) == pytest.approx(1 / 7)
    assert iou(first, touching) == 0


@pytest.mark.parametrize(
    "coords",
    [
        [0, 0, 0, 1000],
        [600, 0, 200, 1000],
        [-1, 0, 1000, 1000],
        [0, 0, 2000, 1000],
        [0, 0, float("nan"), 1000],
    ],
)
def test_invalid_boxes_are_rejected_not_repaired(coords):
    with pytest.raises(ValidationError):
        Box.from_xyxy(coords)


def test_qwen_coordinate_conversion(sample):
    assert GroundingAnswer(bbox_2d=[100, 200, 500, 800]).as_prediction().bbox == sample.target
    with pytest.raises(ValidationError):
        GroundingAnswer(bbox_2d=[100, 200, 50, 800])


def source_row(ref_id=10, image_id=20):
    ann_id = ref_id + 20
    return {
        "ref_id": ref_id,
        "image_id": image_id,
        "ann_id": ann_id,
        "split": "train",
        "sentences": [{"sent": "the red cup"}],
        "bbox": [20, 20, 100, 80],
        "raw_anns": json.dumps({"id": ann_id, "image_id": image_id, "bbox": [20, 20, 80, 60]}),
        "raw_image_info": json.dumps({"id": image_id, "width": 200, "height": 100}),
    }


@pytest.mark.parametrize(
    "count,expected_images,expected_offsets",
    [(2, [20, 21], [0, 2]), (3, [20, 21, 19], [0, 2])],
)
def test_import_selection_uses_first_distinct_images_in_source_order(
    count, expected_images, expected_offsets, monkeypatch
):
    monkeypatch.setattr("dataset.ROW_PAGE_SIZE", 2)
    rows = [source_row(10 + index, image_id) for index, image_id in enumerate([20, 20, 21, 19, 21])]
    offsets = []

    def respond(request):
        offset = int(request.url.params["offset"])
        length = int(request.url.params["length"])
        offsets.append(offset)
        return httpx.Response(
            200,
            json={
                "num_rows_total": len(rows),
                "rows": [{"row": row} for row in rows[offset : offset + length]],
            },
        )

    with HTTPClient(transport=httpx.MockTransport(respond)) as client:
        references = list(iter_references(client, count))
    assert [reference.image_id for reference in references] == expected_images
    assert [reference.ref_id for reference in references] == [10, 12, 13][:count]
    assert offsets == expected_offsets


def test_import_reports_when_too_few_distinct_images_are_available():
    def respond(request):
        return httpx.Response(
            200,
            json={"num_rows_total": 2, "rows": [{"row": source_row()}, {"row": source_row(11)}]},
        )

    with HTTPClient(transport=httpx.MockTransport(respond)) as client:
        with pytest.raises(ValueError, match="Not enough distinct images"):
            list(iter_references(client, 2))


def test_source_pagination_accepts_a_short_final_page():
    rows = [source_row(index, 20) for index in range(100)] + [
        source_row(100, 21),
        source_row(101, 22),
    ]
    offsets = []

    def respond(request):
        offset = int(request.url.params["offset"])
        assert request.url.params["length"] == "100"
        offsets.append(offset)
        return httpx.Response(
            200,
            json={
                "num_rows_total": len(rows),
                "rows": [{"row": row} for row in rows[offset : offset + 100]],
            },
        )

    with HTTPClient(transport=httpx.MockTransport(respond)) as client:
        references = list(iter_references(client, 3))
    assert [reference.image_id for reference in references] == [20, 21, 22]
    assert offsets == [0, 100]


@pytest.mark.parametrize("failure", [429, 503, httpx.ConnectError, httpx.ReadTimeout])
def test_source_retries_temporary_failures(failure, monkeypatch):
    calls, sleeps = [], []
    monkeypatch.setattr("dataset.time.sleep", sleeps.append)

    def respond(request):
        calls.append(request)
        if len(calls) == 1:
            if isinstance(failure, int):
                return httpx.Response(failure, text="private server body")
            raise failure("private server body", request=request)
        return httpx.Response(200, content=b"image bytes")

    with HTTPClient(transport=httpx.MockTransport(respond)) as client:
        result = request_source(client, "https://example.com/image.jpg", "COCO image 20")
    assert result.content == b"image bytes"
    assert len(calls) == 2
    assert sleeps == [2]


def test_source_failure_reports_context_and_stops_after_bounded_retries(monkeypatch, capsys):
    calls, sleeps = [], []
    monkeypatch.setattr("dataset.time.sleep", sleeps.append)

    def respond(request):
        calls.append(request)
        return httpx.Response(502, text="private server body")

    with HTTPClient(transport=httpx.MockTransport(respond)) as client:
        with pytest.raises(
            DatasetError, match="RefCOCO annotations at row 100: HTTP 502 from example.com"
        ):
            request_source(client, "https://example.com/rows", "RefCOCO annotations at row 100")
    assert len(calls) == 4
    assert sleeps == [2, 4, 8]
    assert "private server body" not in capsys.readouterr().out


def test_missing_source_image_is_not_retried(monkeypatch):
    calls = []
    monkeypatch.setattr("dataset.time.sleep", lambda _: pytest.fail("Do not retry HTTP 404"))

    def respond(request):
        calls.append(request)
        return httpx.Response(404)

    with HTTPClient(transport=httpx.MockTransport(respond)) as client:
        with pytest.raises(DatasetError, match="COCO image 20 for reference 10: HTTP 404"):
            request_source(
                client, "https://example.com/image.jpg", "COCO image 20 for reference 10"
            )
    assert len(calls) == 1


@pytest.mark.parametrize(
    "retry_after,expected_sleep", [("5", [5]), ("120", []), ("Thu, 01 Jan 2099 00:00:00 GMT", [])]
)
def test_source_honours_retry_after_without_waiting_too_long(
    retry_after, expected_sleep, monkeypatch
):
    calls, sleeps = [], []
    monkeypatch.setattr("dataset.time.sleep", sleeps.append)

    def respond(request):
        calls.append(request)
        return (
            httpx.Response(200)
            if len(calls) == 2
            else httpx.Response(429, headers={"Retry-After": retry_after})
        )

    with HTTPClient(transport=httpx.MockTransport(respond)) as client:
        if expected_sleep:
            request_source(client, "https://example.com/rows", "RefCOCO annotations")
        else:
            with pytest.raises(DatasetError, match="source asks to wait"):
                request_source(client, "https://example.com/rows", "RefCOCO annotations")
    assert sleeps == expected_sleep


def test_mirror_adapter_reads_raw_coco_annotation():
    row = source_row()
    reference = reference_from_row(row)
    assert reference.target.as_xyxy() == [100, 200, 500, 800]
    assert reference.referring_expressions == ["the red cup"]
    assert reference.image_url.endswith("000000000020.jpg")
    row["image_id"] = 99
    with pytest.raises(ValueError, match="IDs disagree"):
        reference_from_row(row)


@pytest.mark.parametrize("split", ["val", "test", "testA"])
def test_source_evaluation_data_is_not_imported(split):
    row = source_row()
    row["split"] = split
    with pytest.raises(ValueError, match="evaluation"):
        reference_from_row(row)


def test_image_dimensions_checked_in_memory(image_bytes):
    assert load_image(image_bytes, 200, 100).size == (200, 100)
    with pytest.raises(ValueError, match="dimensions"):
        load_image(image_bytes, 999, 100)


def dataset_transport(monkeypatch, image, *, bad_dimensions=False, partial=False, downloads=None):
    row = source_row()
    if bad_dimensions:
        row["raw_image_info"] = json.dumps({"id": 20, "width": 999, "height": 100})
    buffer = BytesIO()
    image.save(buffer, format="JPEG")

    def respond(request):
        if request.url.host == "huggingface.co":
            return httpx.Response(200, json={"sha": "test-revision"})
        if request.url.host == "datasets-server.huggingface.co":
            assert request.url.params["split"] == "train"
            return httpx.Response(
                200,
                json={
                    "num_rows_total": 1,
                    "rows": [{"row": row}],
                    "partial": partial,
                },
            )
        assert request.url.host == "s3.amazonaws.com"
        if downloads is not None:
            downloads.append(request.url.path)
        return httpx.Response(200, content=buffer.getvalue())

    client = HTTPClient(transport=httpx.MockTransport(respond))
    monkeypatch.setattr("dataset.httpx.Client", lambda **kwargs: client)
    return buffer.getvalue()


def test_import_goes_directly_to_cloud_and_reuses_originals(image, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    memory, cloud = MemorySupabase(), MemoryDrive()
    downloads = []
    for index in range(2):
        original = dataset_transport(monkeypatch, image, downloads=downloads)
        report = import_refcoco(cloud.drive, SupabaseStore(memory.client), count=1)
        assert report["sample_rows"] == report["unique_images"] == 1
        assert report["downloaded"] == 1 - index
        assert report["skipped"] == index
    assert len(downloads) == 1
    assert list(tmp_path.iterdir()) == []
    assert len(memory.tables["samples"]) == 1
    assert memory.tables["runs"] == memory.tables["attempts"] == {}
    record = next(iter(memory.tables["samples"].values()))
    assert record["bbox_2d"] == [100, 200, 500, 800]
    assert record["referring_expressions"] == ["the red cup"]
    assert "split" not in record
    assert {
        row["name"] for row in cloud.objects.values() if row.get("parents") == ["root-folder"]
    } == {"samples"}
    assert cloud.contents[record["image_file_id"]] == original
    assert cloud.objects[record["image_file_id"]]["name"] == "sample 1.jpg"
    assert sum(method == "PUT" for method, _ in cloud.requests) == 1


def test_bad_image_never_creates_an_uploaded_image_or_grounding_row(image, monkeypatch):
    memory, cloud = MemorySupabase(), MemoryDrive()
    dataset_transport(monkeypatch, image, bad_dimensions=True)
    with pytest.raises(ValueError, match="dimensions"):
        import_refcoco(cloud.drive, SupabaseStore(memory.client), 1)
    assert cloud.contents == {}
    assert memory.tables["samples"] == {}


def test_partial_source_annotation_never_creates_a_record(image, monkeypatch):
    memory, cloud = MemorySupabase(), MemoryDrive()
    dataset_transport(monkeypatch, image, partial=True)
    with pytest.raises(DatasetError, match="complete rows"):
        import_refcoco(cloud.drive, SupabaseStore(memory.client), 1)
    assert cloud.contents == {}
    assert memory.tables["samples"] == {}


def test_failed_database_import_can_be_repeated_without_uploading_again(image, monkeypatch):
    memory, cloud = MemorySupabase(), MemoryDrive()
    memory.fail_sample_write = True
    downloads = []
    dataset_transport(monkeypatch, image, downloads=downloads)
    with pytest.raises(SupabaseError):
        import_refcoco(cloud.drive, SupabaseStore(memory.client), 1)
    assert memory.tables["samples"] == {}
    memory.fail_sample_write = False
    dataset_transport(monkeypatch, image, downloads=downloads)
    report = import_refcoco(
        GoogleDrive(cloud.service, "root-folder"), SupabaseStore(memory.client), 1
    )
    assert len(memory.tables["samples"]) == 1
    assert report["downloaded"] == 0
    assert len(downloads) == 1
    assert sum(method == "PUT" for method, _ in cloud.requests) == 1


def test_source_failure_keeps_saved_samples_and_rerun_completes_without_duplicates(
    image, monkeypatch
):
    memory, cloud = MemorySupabase(), MemoryDrive()
    rows = [source_row(10, 20), source_row(11, 21)]
    originals = {}
    for image_id, colour in [(20, "white"), (21, "red")]:
        buffer = BytesIO()
        original = image.copy()
        original.paste(colour, (0, 0, image.width, image.height))
        original.save(buffer, format="JPEG")
        originals[image_id] = buffer.getvalue()
    missing_second = True
    downloads = []

    def respond(request):
        if request.url.host == "huggingface.co":
            return httpx.Response(200, json={"sha": "test-revision"})
        if request.url.host == "datasets-server.huggingface.co":
            return httpx.Response(
                200, json={"num_rows_total": 2, "rows": [{"row": row} for row in rows]}
            )
        image_id = int(request.url.path.rsplit("_", 1)[1].removesuffix(".jpg"))
        downloads.append(image_id)
        if image_id == 21 and missing_second:
            return httpx.Response(404)
        return httpx.Response(200, content=originals[image_id])

    monkeypatch.setattr(
        "dataset.httpx.Client", lambda **kwargs: HTTPClient(transport=httpx.MockTransport(respond))
    )
    with pytest.raises(DatasetError, match="COCO image 21 for reference 11: HTTP 404"):
        import_refcoco(cloud.drive, SupabaseStore(memory.client), 2)
    assert len(memory.tables["samples"]) == len(cloud.contents) == 1
    missing_second = False
    another_client = GoogleDrive(cloud.service, "root-folder")
    report = import_refcoco(another_client, SupabaseStore(memory.client), 2)
    assert report["sample_rows"] == report["unique_images"] == 2
    assert len(memory.tables["samples"]) == len(cloud.contents) == 2
    assert report["skipped"] == report["downloaded"] == 1
    assert downloads == [20, 21, 21]
    assert sum(method == "PUT" for method, _ in cloud.requests) == 2
    assert {cloud.objects[file_id]["name"] for file_id in cloud.contents} == {
        "sample 1.jpg",
        "sample 2.jpg",
    }


@pytest.mark.parametrize("missing", ["deleted", "trashed"])
def test_import_repairs_missing_images_without_overwriting_labels(image, monkeypatch, missing):
    memory, cloud = MemorySupabase(), MemoryDrive()
    downloads = []
    dataset_transport(monkeypatch, image, downloads=downloads)
    import_refcoco(cloud.drive, SupabaseStore(memory.client), 1)
    row = next(iter(memory.tables["samples"].values()))
    old_file_id = row["image_file_id"]
    row["bbox_2d"] = [90, 180, 540, 850]
    row["referring_expressions"] = ["a manually edited expression"]
    labels = dict(row)
    if missing == "deleted":
        del cloud.objects[old_file_id]
        del cloud.contents[old_file_id]
    else:
        cloud.objects[old_file_id]["trashed"] = True
    dataset_transport(monkeypatch, image, downloads=downloads)
    report = import_refcoco(
        GoogleDrive(cloud.service, "root-folder"), SupabaseStore(memory.client), 1
    )
    assert report["downloaded"] == 1 and report["skipped"] == 0
    assert len(downloads) == 2
    assert len(memory.tables["samples"]) == 1
    assert row["image_file_id"] != old_file_id
    assert row == {**labels, "image_file_id": row["image_file_id"]}
    assert cloud.drive.image_exists(row["image_file_id"])


def test_drive_permission_error_stops_before_an_image_download(image, monkeypatch):
    memory, cloud = MemorySupabase(), MemoryDrive()
    dataset_transport(monkeypatch, image)
    import_refcoco(cloud.drive, SupabaseStore(memory.client), 1)
    file_id = next(iter(memory.tables["samples"].values()))["image_file_id"]
    cloud.file_errors[file_id] = 403
    downloads = []
    dataset_transport(monkeypatch, image, downloads=downloads)
    with pytest.raises(DriveError, match="HTTP 403"):
        import_refcoco(GoogleDrive(cloud.service, "root-folder"), SupabaseStore(memory.client), 1)
    assert downloads == []
    assert len(cloud.contents) == len(memory.tables["samples"]) == 1


def test_older_saved_sample_is_skipped_without_a_source_identifier(image, monkeypatch):
    memory, cloud = MemorySupabase(), MemoryDrive()
    original = dataset_transport(monkeypatch, image)
    file_id = cloud.drive.upload("samples/sample 1.jpg", original, "image/jpeg", immutable=True)
    source = reference_from_row(source_row())
    memory.seed(source.sample_record(file_id))
    downloads = []
    dataset_transport(monkeypatch, image, downloads=downloads)
    report = import_refcoco(
        GoogleDrive(cloud.service, "root-folder"), SupabaseStore(memory.client), 1
    )
    assert report["skipped"] == 1 and report["downloaded"] == 0
    assert downloads == []
    assert "properties" not in cloud.objects[file_id]
    assert all(request.method == "GET" for request in memory.requests)
