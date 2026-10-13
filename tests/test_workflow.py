import json
from datetime import UTC, datetime, timedelta
from io import BytesIO
from uuid import UUID

import pytest
from PIL import Image
from test_agent import ScriptedRoles, wrong_box
from test_google_drive import MemoryDrive
from test_supabase import MemorySupabase

from dataset import DatasetError
from google_drive import DriveError
from main import main, run_batch
from model_client import ModelError
from models import Prediction, Settings
from outputs import export_training
from supabase_client import SupabaseError, SupabaseStore


def cloud_for(sample_record, image_bytes):
    memory, cloud = MemorySupabase(), MemoryDrive()
    memory.seed(sample_record)
    cloud.put_original(sample_record.image_file_id, image_bytes)
    return memory, cloud, SupabaseStore(memory.client)


def configure_cli(monkeypatch, memory, cloud):
    monkeypatch.setattr("main.create_supabase_client", lambda: memory.client)
    monkeypatch.setattr("main.create_drive_client", lambda: cloud.drive)


def test_cloud_loop_saves_original_boxes_attempt_image_and_stored_output(
    sample_record, sample, image_bytes, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    memory, cloud, store = cloud_for(sample_record, image_bytes)
    roles = ScriptedRoles(sample, [wrong_box()], settings=Settings(max_rounds=2))
    roles.strong_solver.solve = lambda *args: Prediction(
        bbox=sample.target.model_copy(update={"x_min": 110})
    )
    summary = run_batch(store.iter_samples(1), 1, roles.main, store, cloud.drive)
    assert summary["accepted"] == 1
    assert list(tmp_path.iterdir()) == []
    attempt = next(iter(memory.tables["attempts"].values()))
    assert attempt["strong_bbox_2d"][0] == 110
    accepted = memory.outputs_for(store.run_id)
    assert accepted[0]["bbox_2d"] == [100, 200, 500, 800]
    assert accepted[0]["image_file_id"] == sample.image_file_id
    assert accepted[0]["attempt_id"] == attempt["id"]
    assert "run_id" not in accepted[0] and "sample_id" not in accepted[0]
    assert cloud.contents[sample.image_file_id] == image_bytes
    assert cloud.objects[attempt["attempt_file_id"]]["name"] == "sample 1 attempt 1.png"
    with Image.open(BytesIO(cloud.contents[attempt["attempt_file_id"]])) as preview:
        left = (preview.width - sample.width) // 2
        marked = preview.crop(
            (left, preview.height - sample.height, left + sample.width, preview.height)
        )
        assert marked.getpixel((40, 25)) == (0, 255, 0)
        assert marked.getpixel((140, 10)) == (255, 0, 0)
        assert marked.getpixel((40, 20)) == (0, 0, 255)
    assert memory.tables["runs"][store.run_id]["status"] == "completed"
    assert memory.tables["runs"][store.run_id]["finished_at"]
    assert sum(method == "PUT" for method, _ in cloud.requests) == 1
    folders = [
        row
        for row in cloud.objects.values()
        if row.get("mimeType") == "application/vnd.google-apps.folder"
    ]
    assert {row["name"] for row in folders if row.get("parents") == ["root-folder"]} == {"attempts"}
    training_dir = tmp_path / "training"
    assert export_training(store, cloud.drive, store.run_id, training_dir) == 1
    training = json.loads((training_dir / "qwen_train.json").read_text())
    assert training[0]["conversations"][1]["value"] == '{"bbox_2d": [100, 200, 500, 800]}'
    assert (training_dir / training[0]["images"][0]).read_bytes() == image_bytes


def test_rejected_and_accepted_attempts_have_separate_previews(sample_record, sample, image_bytes):
    memory, cloud, store = cloud_for(sample_record, image_bytes)
    roles = ScriptedRoles(sample, [sample.target, wrong_box()], settings=Settings(max_rounds=3))
    run_batch(store.iter_samples(1), 1, roles.main, store, cloud.drive)
    attempts = list(memory.tables["attempts"].values())
    assert [attempt["status"] for attempt in attempts] == ["rejected", "accepted"]
    assert attempts[0]["weak_iou"] == attempts[0]["strong_iou"] == 1
    assert attempts[1]["weak_iou"] == 0
    assert len({attempt["attempt_file_id"] for attempt in attempts}) == 2
    assert len(memory.outputs_for(store.run_id)) == 1


def test_attempt_names_use_original_number_and_do_not_overwrite_shared_images(
    sample_record, sample, image_bytes
):
    memory, cloud, store = cloud_for(sample_record, image_bytes)
    cloud.objects[sample_record.image_file_id]["name"] = "sample 7.png"
    memory.seed(sample_record.model_copy(update={"id": UUID(int=2)}))
    roles = ScriptedRoles(sample, [sample.target, wrong_box(), wrong_box()])
    run_batch(store.iter_samples(2), 2, roles.main, store, cloud.drive)
    attempts = list(memory.tables["attempts"].values())
    assert [cloud.objects[row["attempt_file_id"]]["name"] for row in attempts] == [
        "sample 7 attempt 1.png",
        "sample 7 attempt 2.png",
        "sample 7 attempt 3.png",
    ]
    assert len({row["attempt_file_id"] for row in attempts}) == 3
    assert [row["attempt_number"] for row in attempts] == [1, 2, 1]


def test_zero_acceptance_keeps_attempt_and_outputs_empty(
    sample_record, sample, image_bytes, tmp_path
):
    memory, cloud, store = cloud_for(sample_record, image_bytes)
    roles = ScriptedRoles(sample, settings=Settings(max_rounds=1))
    summary = run_batch(store.iter_samples(1), 1, roles.main, store, cloud.drive)
    assert summary["exhausted"] == 1
    assert memory.outputs_for(store.run_id) == []
    assert "accepted_file_id" not in summary
    assert next(iter(memory.tables["attempts"].values()))["status"] == "rejected"
    with pytest.raises(ValueError, match="No accepted"):
        export_training(store, cloud.drive, store.run_id, tmp_path / "training")
    assert list(tmp_path.iterdir()) == []


def test_model_failure_keeps_partial_attempt_and_previous_acceptance(
    sample_record, sample, image_bytes
):
    memory, cloud, store = cloud_for(sample_record, image_bytes)
    second = sample_record.model_copy(
        update={
            "id": UUID("00000000-0000-0000-0000-000000000002"),
            "referring_expressions": ["another reference"],
        }
    )
    memory.seed(second)
    roles = ScriptedRoles(sample, [wrong_box()], settings=Settings(max_rounds=1))
    generate = roles.challenger.generate

    def fail_on_second(image, target, direction, history):
        if target.ref_id == 2:
            raise ModelError("Challenger unavailable")
        return generate(image, target, direction, history)

    roles.challenger.generate = fail_on_second
    summary = run_batch(store.iter_samples(2), 2, roles.main, store, cloud.drive)
    assert summary["accepted"] == summary["errors"] == 1
    attempts = list(memory.tables["attempts"].values())
    assert [attempt["status"] for attempt in attempts] == ["accepted", "error"]
    assert attempts[-1]["referring_expression"] is None
    assert attempts[-1]["weak_bbox_2d"] is None
    assert attempts[-1]["feedback"] == "Challenger unavailable"
    assert attempts[-1]["attempt_file_id"]
    assert memory.tables["runs"][store.run_id]["status"] == "error"
    assert len(memory.outputs_for(store.run_id)) == 1
    downloads = [uri for method, uri in cloud.requests if method == "GET" and "alt=media" in uri]
    assert len(downloads) == 1


@pytest.mark.parametrize("failure", ["image", "database", "output"])
def test_cloud_failure_stops_without_local_fallback(
    sample_record, sample, image_bytes, tmp_path, monkeypatch, failure
):
    monkeypatch.chdir(tmp_path)
    memory, cloud, store = cloud_for(sample_record, image_bytes)
    cloud.fail_upload = failure == "image"
    memory.fail_attempt_write = failure == "database"
    memory.fail_output_write = failure == "output"
    roles = ScriptedRoles(sample, [wrong_box()])
    with pytest.raises((SupabaseError, DriveError)):
        run_batch(store.iter_samples(1), 1, roles.main, store, cloud.drive)
    assert memory.tables["runs"][store.run_id]["status"] == "error"
    assert list(tmp_path.iterdir()) == []
    assert memory.outputs_for(store.run_id) == []
    assert all(row["status"] != "accepted" for row in memory.tables["attempts"].values())


def test_database_loading_paginates_and_keeps_objects_paired(sample_record):
    memory = MemorySupabase()
    for index in range(105):
        sample_id = UUID(int=105 - index)
        memory.seed(
            sample_record.model_copy(
                update={
                    "id": sample_id,
                    "referring_expressions": [f"object {index}"],
                    "bbox_2d": [index, 200, 500, 800],
                }
            )
        )
        memory.tables["samples"][str(sample_id)]["created_at"] = (
            datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=index)
        ).isoformat()
    memory.seed(sample_record.model_copy(update={"id": UUID(int=106), "dataset": "other"}))
    store = SupabaseStore(memory.client)
    assert store.sample_count() == 106
    assert store.sample_count("jxu124/refcoco") == 105
    samples = list(store.iter_samples(101, offset=4, dataset="jxu124/refcoco"))
    assert len(samples) == 101
    assert samples[0].referring_expressions == ["object 4"]
    assert samples[0].target.x_min == 4
    assert samples[-1].referring_expressions == ["object 104"]
    reads = [r for r in memory.requests if r.url.params.get("select", "").startswith("id,image")]
    assert [int(r.url.params["limit"]) for r in reads] == [50, 50, 1]
    assert all("split" not in sample.model_dump() for sample in samples)


def test_import_timestamp_ties_have_stable_order(sample_record):
    memory = MemorySupabase()
    for number in (3, 1, 2):
        sample_id = UUID(int=number)
        memory.seed(sample_record.model_copy(update={"id": sample_id}))
        memory.tables["samples"][str(sample_id)]["created_at"] = "2026-01-01T00:00:00+00:00"
    store = SupabaseStore(memory.client)
    assert [sample.sample_id for sample in store.iter_samples(2, offset=1)] == [
        UUID(int=2),
        UUID(int=3),
    ]


def test_outputs_are_stored_snapshots_and_run_filter_uses_attempt_relation(
    sample_record, sample, image_bytes
):
    memory, cloud, first_store = cloud_for(sample_record, image_bytes)
    roles = ScriptedRoles(sample, [wrong_box()])
    run_batch(first_store.iter_samples(1), 1, roles.main, first_store, cloud.drive)
    first_output = list(first_store.iter_outputs(first_store.run_id))[0]
    memory.tables["samples"][str(sample_record.id)]["bbox_2d"] = [10, 20, 600, 900]
    assert list(first_store.iter_outputs(first_store.run_id))[0] == first_output

    memory.seed(sample_record)
    second_store = SupabaseStore(memory.client)
    roles = ScriptedRoles(sample, [wrong_box()])
    run_batch(second_store.iter_samples(1), 1, roles.main, second_store, cloud.drive)
    second_output = list(second_store.iter_outputs(second_store.run_id))[0]
    assert second_output.attempt_id != first_output.attempt_id
    assert list(first_store.iter_outputs(first_store.run_id)) == [first_output]
    assert len(memory.tables["outputs"]) == 2
    requests = [r for r in memory.requests if r.url.path.endswith("/outputs")]
    assert all("attempts!inner(run_id)" in r.url.params["select"] for r in requests)
    assert all(
        "run_id" not in row and "sample_id" not in row for row in memory.tables["outputs"].values()
    )


def test_import_command_does_not_initialise_model_providers(
    sample_record, image_bytes, monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    memory, cloud, _ = cloud_for(sample_record, image_bytes)
    configure_cli(monkeypatch, memory, cloud)

    def import_dataset(drive, store, count):
        assert drive is cloud.drive
        assert count == 1
        return {"sample_rows": 1, "unique_images": 1}

    monkeypatch.setattr("main.import_refcoco", import_dataset)

    def unexpected(*args, **kwargs):
        pytest.fail("Importer must not initialise model providers")

    monkeypatch.setattr("main.OpenAIClient", unexpected)
    monkeypatch.setattr("main.WeakSolver", unexpected)
    assert main(["import-dataset", "--count", "1"]) == 0
    assert memory.tables["runs"] == memory.tables["attempts"] == {}
    assert list(tmp_path.iterdir()) == []


def test_import_command_reports_the_failed_source(sample_record, image_bytes, monkeypatch, capsys):
    memory, cloud, _ = cloud_for(sample_record, image_bytes)
    configure_cli(monkeypatch, memory, cloud)

    def fail_import(*args):
        raise DatasetError("COCO image 20 for reference 10: HTTP 404 from s3.amazonaws.com.")

    monkeypatch.setattr("main.import_refcoco", fail_import)
    assert main(["import-dataset", "--count", "100"]) == 1
    message = capsys.readouterr().err
    assert "COCO image 20 for reference 10: HTTP 404" in message
    assert "Dataset request failed" not in message


def test_dry_run_reads_cloud_images_without_models_or_writes(
    sample_record, image_bytes, monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    memory, cloud, _ = cloud_for(sample_record, image_bytes)
    configure_cli(monkeypatch, memory, cloud)

    def unexpected(*args, **kwargs):
        pytest.fail("Dry run must not initialise model providers")

    monkeypatch.setattr("main.OpenAIClient", unexpected)
    assert main(["run", "--limit", "1", "--dry-run"]) == 0
    assert memory.tables["runs"] == memory.tables["attempts"] == {}
    assert all(method == "GET" for method, _ in cloud.requests)
    assert list(tmp_path.iterdir()) == []


def test_empty_database_fails_before_model_calls(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    memory, cloud = MemorySupabase(), MemoryDrive()
    configure_cli(monkeypatch, memory, cloud)
    monkeypatch.setattr("main.OpenAIClient", lambda *args: pytest.fail("No data available"))
    assert main(["run", "--limit", "1"]) == 1
    assert "Import more data" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []


def test_local_training_export_paginates_and_downloads_original_once(
    sample_record, image_bytes, tmp_path
):
    memory, cloud, store = cloud_for(sample_record, image_bytes)
    rows = [
        {
            "attempt_id": str(UUID(int=index + 1)),
            "image_file_id": sample_record.image_file_id,
            "width": 200,
            "height": 100,
            "referring_expression": f"the cup, example {index}",
            "bbox_2d": [100, 200, 500, 800],
        }
        for index in range(103)
    ]
    memory.outputs_for = lambda _: rows
    assert export_training(store, cloud.drive, store.run_id, tmp_path / "training") == 103
    downloads = [uri for method, uri in cloud.requests if method == "GET" and "alt=media" in uri]
    assert len(downloads) == 1


def test_training_export_rejects_bad_ids_before_any_file_write(
    sample_record, image_bytes, tmp_path
):
    memory, cloud, store = cloud_for(sample_record, image_bytes)
    memory.outputs_for = lambda _: [
        {
            "attempt_id": str(UUID(int=1)),
            "image_file_id": "../outside",
            "width": 200,
            "height": 100,
            "referring_expression": "the cup",
            "bbox_2d": [100, 200, 500, 800],
        }
    ]
    with pytest.raises(DriveError):
        export_training(store, cloud.drive, store.run_id, tmp_path / "training")
    assert list(tmp_path.iterdir()) == []
