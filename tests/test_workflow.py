import json
from datetime import UTC, datetime, timedelta
from io import BytesIO
from uuid import UUID

import pytest
from conftest import TEST_MODELS, make_settings
from PIL import Image
from test_agent import ScriptedRoles, wrong_box
from test_google_drive import MemoryDrive
from test_supabase import MemorySupabase

import dataset
from dataset import DatasetError
from google_drive import DriveError
from main import main, role_models, run_batch
from model_client import ModelError
from models import Claim, Prediction, Verdict
from outputs import export_training
from solver import grounding_prompt
from supabase_client import SupabaseError, SupabaseStore


def cloud_for(sample_record, image_bytes):
    memory, cloud = MemorySupabase(), MemoryDrive()
    memory.seed(sample_record)
    cloud.put_original(sample_record.image_file_id, image_bytes)
    return memory, cloud, SupabaseStore(memory.client)


def accepted_rounds(memory):
    return [row for row in memory.tables["rounds"].values() if row["status"] == "accepted"]


def add_accepted_round(memory, store, record, number, expression):
    key = (store.run_id, record.ref_id, number)
    memory.tables["rounds"][key] = {
        "run_id": store.run_id,
        "ref_id": record.ref_id,
        "round_number": number,
        "referring_expression": expression,
        "status": "accepted",
    }


def configure_cli(monkeypatch, memory, cloud, module="main"):
    for role, model in TEST_MODELS.items():
        monkeypatch.setenv(f"AUTODATA_{role.removesuffix('_model').upper()}_MODEL", model)
    monkeypatch.setattr(f"{module}.create_supabase_client", lambda: memory.client)
    monkeypatch.setattr(f"{module}.create_drive_client", lambda: cloud.drive)


def test_cloud_loop_saves_original_boxes_round_preview_and_accepted_round(
    sample_record, sample, image_bytes, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    memory, cloud, store = cloud_for(sample_record, image_bytes)
    roles = ScriptedRoles(sample, [wrong_box()], settings=make_settings(max_rounds=2))
    roles.strong_solver.solve = lambda *args: Prediction(
        bbox=sample.target.model_copy(update={"x_min": 110})
    )
    summary = run_batch(store.iter_samples(1), 1, roles.main, store, cloud.drive)
    assert summary["accepted"] == 1
    assert list(tmp_path.iterdir()) == []
    row = next(iter(memory.tables["rounds"].values()))
    assert row["strong_bbox_2d"][0] == 110
    assert row["status"] == "accepted"
    example = next(store.iter_accepted_examples([store.run_id]))
    assert example.bbox_2d == [100, 200, 500, 800]
    assert example.image_file_id == sample.image_file_id
    assert example.referring_expression == row["referring_expression"]
    assert cloud.contents[sample.image_file_id] == image_bytes
    assert cloud.objects[row["preview_file_id"]]["name"] == f"{store.run_id} ref 1 round 1.png"
    with Image.open(BytesIO(cloud.contents[row["preview_file_id"]])) as preview:
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
    assert {row["name"] for row in folders if row.get("parents") == ["root-folder"]} == {
        "images",
        "previews",
    }
    training_dir = tmp_path / "training"
    assert export_training(store, cloud.drive, [store.run_id], training_dir) == 1
    training = json.loads((training_dir / "qwen_train.json").read_text())
    assert training[0]["conversations"][0]["value"] == "<image>\n" + grounding_prompt(
        row["referring_expression"]
    )
    assert training[0]["conversations"][1]["value"] == '{"bbox_2d": [100, 200, 500, 800]}'
    assert (training_dir / training[0]["images"][0]).read_bytes() == image_bytes


def test_rejected_and_accepted_rounds_have_separate_previews(sample_record, sample, image_bytes):
    memory, cloud, store = cloud_for(sample_record, image_bytes)
    roles = ScriptedRoles(
        sample, [sample.target, wrong_box()], settings=make_settings(max_rounds=3)
    )
    run_batch(store.iter_samples(1), 1, roles.main, store, cloud.drive)
    saved = list(memory.tables["rounds"].values())
    assert [row["status"] for row in saved] == ["rejected", "accepted"]
    assert saved[0]["weak_iou"] == saved[0]["strong_iou"] == 1
    assert saved[1]["weak_iou"] == 0
    assert len({row["preview_file_id"] for row in saved}) == 2
    assert len(accepted_rounds(memory)) == 1


def test_previews_are_named_by_ref_id_and_do_not_overwrite_shared_images(
    sample_record, sample, image_bytes
):
    memory, cloud, store = cloud_for(sample_record, image_bytes)
    memory.seed(sample_record.model_copy(update={"ref_id": 2}))
    roles = ScriptedRoles(sample, [sample.target, wrong_box(), wrong_box()])
    run_batch(store.iter_samples(2), 2, roles.main, store, cloud.drive)
    saved = list(memory.tables["rounds"].values())
    assert [cloud.objects[row["preview_file_id"]]["name"] for row in saved] == [
        f"{store.run_id} ref 1 round 1.png",
        f"{store.run_id} ref 1 round 2.png",
        f"{store.run_id} ref 2 round 1.png",
    ]
    assert len({row["preview_file_id"] for row in saved}) == 3
    assert [row["round_number"] for row in saved] == [1, 2, 1]
    downloads = [uri for method, uri in cloud.requests if method == "GET" and "alt=media" in uri]
    assert len(downloads) == 1


def test_no_accepted_round_keeps_rounds_rejected_and_export_empty(
    sample_record, sample, image_bytes, tmp_path
):
    memory, cloud, store = cloud_for(sample_record, image_bytes)
    roles = ScriptedRoles(sample, settings=make_settings(max_rounds=1))
    summary = run_batch(store.iter_samples(1), 1, roles.main, store, cloud.drive)
    assert summary["exhausted"] == 1
    assert accepted_rounds(memory) == []
    assert "accepted_file_id" not in summary
    assert next(iter(memory.tables["rounds"].values()))["status"] == "rejected"
    with pytest.raises(ValueError, match="No accepted"):
        export_training(store, cloud.drive, [store.run_id], tmp_path / "training")
    assert list(tmp_path.iterdir()) == []


def test_judge_rejecting_every_round_saves_rejected_rows_and_exhausts(
    sample_record, sample, image_bytes
):
    memory, cloud, store = cloud_for(sample_record, image_bytes)
    roles = ScriptedRoles(sample, [wrong_box(), wrong_box()], settings=make_settings(max_rounds=2))
    roles.verdict = Verdict(
        claims=[Claim(claim="purple apple", visible=False)], other_matches=0, reason="Red apple."
    )
    summary = run_batch(store.iter_samples(1), 1, roles.main, store, cloud.drive)
    assert summary["exhausted"] == 1
    saved = list(memory.tables["rounds"].values())
    assert len(saved) == 2
    assert all(row["status"] == "rejected" for row in saved)
    assert all(row["feedback"].startswith("Rejected by judge") for row in saved)
    assert accepted_rounds(memory) == []
    run_row = next(iter(memory.tables["runs"].values()))
    assert run_row["judge_model"] == make_settings().judge_model


def test_model_failure_is_recorded_and_the_batch_continues(sample_record, sample, image_bytes):
    memory, cloud, store = cloud_for(sample_record, image_bytes)
    second = sample_record.model_copy(
        update={
            "referring_expressions": ["another reference"],
            "ref_id": 2,
        }
    )
    memory.seed(second)
    roles = ScriptedRoles(sample, [wrong_box()], settings=make_settings(max_rounds=1))
    generate = roles.challenger.generate

    def fail_on_first(image, target, direction, history):
        if target.ref_id == 1:
            raise ModelError("Challenger unavailable")
        return generate(image, target, direction, history)

    roles.challenger.generate = fail_on_first
    summary = run_batch(store.iter_samples(2), 2, roles.main, store, cloud.drive)
    assert summary["errors"] == summary["accepted"] == 1
    assert summary["status"] == "completed"
    saved = list(memory.tables["rounds"].values())
    assert [row["status"] for row in saved] == ["error", "accepted"]
    assert saved[0]["referring_expression"] is None
    assert saved[0]["weak_bbox_2d"] is None
    assert saved[0]["feedback"] == "Challenger unavailable"
    assert saved[0]["preview_file_id"]
    assert memory.tables["runs"][store.run_id]["status"] == "completed"
    assert len(accepted_rounds(memory)) == 1
    downloads = [uri for method, uri in cloud.requests if method == "GET" and "alt=media" in uri]
    assert len(downloads) == 1


def test_three_consecutive_failed_targets_stop_the_batch(sample_record, sample, image_bytes):
    memory, cloud, store = cloud_for(sample_record, image_bytes)
    for ref_id in (2, 3, 4):
        memory.seed(sample_record.model_copy(update={"ref_id": ref_id}))
    roles = ScriptedRoles(sample, settings=make_settings(max_rounds=1))

    def always_fail(image, target, direction, history):
        raise ModelError("Challenger unavailable")

    roles.challenger.generate = always_fail
    summary = run_batch(store.iter_samples(4), 4, roles.main, store, cloud.drive)
    assert summary["processed"] == summary["errors"] == 3
    assert summary["unprocessed"] == 1
    assert summary["status"] == "error"
    assert memory.tables["runs"][store.run_id]["status"] == "error"
    saved = list(memory.tables["rounds"].values())
    assert [row["status"] for row in saved] == ["error"] * 3


@pytest.mark.parametrize("failure", ["image", "database"])
def test_cloud_failure_stops_without_local_fallback(
    sample_record, sample, image_bytes, tmp_path, monkeypatch, failure
):
    monkeypatch.chdir(tmp_path)
    memory, cloud, store = cloud_for(sample_record, image_bytes)
    cloud.fail_upload = failure == "image"
    memory.fail_round_write = failure == "database"
    roles = ScriptedRoles(sample, [wrong_box()])
    with pytest.raises((SupabaseError, DriveError)):
        run_batch(store.iter_samples(1), 1, roles.main, store, cloud.drive)
    assert memory.tables["runs"][store.run_id]["status"] == "error"
    assert list(tmp_path.iterdir()) == []
    assert accepted_rounds(memory) == []


def test_database_loading_paginates_and_keeps_objects_paired(sample_record):
    memory = MemorySupabase()
    for index in range(105):
        ref_id = 105 - index
        memory.seed(
            sample_record.model_copy(
                update={
                    "ref_id": ref_id,
                    "referring_expressions": [f"object {index}"],
                    "bbox_2d": [index, 200, 500, 800],
                }
            )
        )
        memory.tables["refcoco_annotations"][ref_id]["created_at"] = (
            datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=index)
        ).isoformat()
    store = SupabaseStore(memory.client)
    assert store.annotation_count() == 105
    samples = list(store.iter_samples(101, offset=4))
    assert len(samples) == 101
    assert samples[0].referring_expressions == ["object 4"]
    assert samples[0].target.x_min == 4
    assert samples[-1].referring_expressions == ["object 104"]
    reads = [
        r for r in memory.requests if r.url.params.get("select", "").startswith("ref_id,image")
    ]
    assert [int(r.url.params["limit"]) for r in reads] == [50, 50, 1]
    assert all("split" not in sample.model_dump() for sample in samples)


def test_import_timestamp_ties_have_stable_order(sample_record):
    memory = MemorySupabase()
    for number in (3, 1, 2):
        memory.seed(sample_record.model_copy(update={"ref_id": number}))
        memory.tables["refcoco_annotations"][number]["created_at"] = "2026-01-01T00:00:00+00:00"
    store = SupabaseStore(memory.client)
    assert [sample.ref_id for sample in store.iter_samples(2, offset=1)] == [2, 3]


def test_export_reads_only_the_runs_accepted_rounds_joined_to_their_annotation(
    sample_record, sample, image_bytes
):
    memory, cloud, first_store = cloud_for(sample_record, image_bytes)
    roles = ScriptedRoles(sample, [wrong_box()])
    run_batch(first_store.iter_samples(1), 1, roles.main, first_store, cloud.drive)
    second_store = SupabaseStore(memory.client)
    roles = ScriptedRoles(sample, [wrong_box()])
    run_batch(second_store.iter_samples(1), 1, roles.main, second_store, cloud.drive)
    assert len(accepted_rounds(memory)) == 2
    first = list(first_store.iter_accepted_examples([first_store.run_id]))
    assert len(first) == 1
    assert first[0].image_file_id == sample.image_file_id
    assert first[0].bbox_2d == [100, 200, 500, 800]
    assert (first[0].width, first[0].height) == (sample.width, sample.height)
    other = list(second_store.iter_accepted_examples([second_store.run_id]))
    assert len(other) == 1
    assert list(SupabaseStore(memory.client).iter_accepted_examples([str(UUID(int=99))])) == []
    both = list(first_store.iter_accepted_examples([first_store.run_id, second_store.run_id]))
    assert len(both) == 2


def test_import_command_does_not_initialise_model_providers(
    sample_record, image_bytes, monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    memory, cloud, _ = cloud_for(sample_record, image_bytes)
    configure_cli(monkeypatch, memory, cloud, "dataset")

    def import_dataset(drive, store, count):
        assert drive is cloud.drive
        assert count == 1
        return {"sample_rows": 1, "unique_images": 1}

    monkeypatch.setattr("dataset.import_refcoco", import_dataset)

    def unexpected(*args, **kwargs):
        pytest.fail("Importer must not initialise model providers")

    monkeypatch.setattr("main.OpenAIClient", unexpected)
    monkeypatch.setattr("main.OpenRouterClient", unexpected)
    monkeypatch.setattr("main.Solver", unexpected)
    assert dataset.main(["--count", "1"]) == 0
    assert memory.tables["runs"] == memory.tables["rounds"] == {}
    assert list(tmp_path.iterdir()) == []


def test_import_command_reports_the_failed_source(sample_record, image_bytes, monkeypatch, capsys):
    memory, cloud, _ = cloud_for(sample_record, image_bytes)
    configure_cli(monkeypatch, memory, cloud, "dataset")

    def fail_import(*args):
        raise DatasetError("COCO image 20 for reference 10: HTTP 404 from s3.amazonaws.com.")

    monkeypatch.setattr("dataset.import_refcoco", fail_import)
    assert dataset.main(["--count", "100"]) == 1
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
    monkeypatch.setattr("main.OpenRouterClient", unexpected)
    assert main(["run", "--limit", "1", "--dry-run"]) == 0
    assert memory.tables["runs"] == memory.tables["rounds"] == {}
    assert all(method == "GET" for method, _ in cloud.requests)
    assert list(tmp_path.iterdir()) == []


def test_empty_database_fails_before_model_calls(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    memory, cloud = MemorySupabase(), MemoryDrive()
    configure_cli(monkeypatch, memory, cloud)
    monkeypatch.setattr("main.OpenAIClient", lambda *args: pytest.fail("No data available"))
    monkeypatch.setattr("main.OpenRouterClient", lambda *args: pytest.fail("No data available"))
    assert main(["run", "--limit", "1"]) == 1
    assert "Import more data" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []


def test_local_training_export_paginates_and_downloads_original_once(
    sample_record, image_bytes, tmp_path
):
    memory, cloud, store = cloud_for(sample_record, image_bytes)
    for index in range(103):
        add_accepted_round(memory, store, sample_record, index + 1, f"the cup, example {index}")
    assert export_training(store, cloud.drive, [store.run_id], tmp_path / "training") == 103
    downloads = [uri for method, uri in cloud.requests if method == "GET" and "alt=media" in uri]
    assert len(downloads) == 1


def test_training_export_rejects_bad_ids_before_any_file_write(
    sample_record, image_bytes, tmp_path
):
    memory, cloud, store = cloud_for(sample_record, image_bytes)
    memory.tables["refcoco_annotations"][sample_record.ref_id]["image_file_id"] = "../outside"
    add_accepted_round(memory, store, sample_record, 1, "the cup")
    with pytest.raises(DriveError):
        export_training(store, cloud.drive, [store.run_id], tmp_path / "training")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("provider", ["openrouter", "openai"])
def test_role_models_routes_strong_solver_by_provider(provider):
    openai_client, openrouter = object(), object()
    settings = make_settings(strong_provider=provider)
    roles = role_models(settings, openai_client, openrouter)
    assert list(roles) == ["main_agent", "challenger", "strong_solver", "weak_solver", "judge"]
    strong_client = openrouter if provider == "openrouter" else openai_client
    assert roles["strong_solver"] == (strong_client, settings.strong_model)
    assert roles["main_agent"] == (openai_client, settings.main_model)
    assert roles["challenger"] == (openai_client, settings.challenger_model)
    assert roles["weak_solver"] == (openrouter, settings.weak_model)
    assert roles["judge"] == (openrouter, settings.judge_model)
