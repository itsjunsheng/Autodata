# Import cloud grounding data, run the agent loop, and export accepted examples.

import argparse
import json
import sys
from collections import Counter
from collections.abc import Iterable
from io import BytesIO
from pathlib import Path
from uuid import UUID

import httpx
from dotenv import load_dotenv

from challenger import Challenger
from dataset import load_image
from google_drive import (
    PREVIEWS_FOLDER,
    DriveError,
    GoogleDrive,
    authenticate_drive,
    create_drive_client,
)
from judge import Judge
from main_agent import MainAgent
from model_client import ModelClient, ModelError, OpenAIClient, OpenRouterClient
from models import Result, Sample, Settings
from outputs import export_training, render_round_preview, round_status
from solver import Solver
from supabase_client import (
    SupabaseError,
    SupabaseStore,
    check_supabase_connection,
    create_supabase_client,
)
from verifier import Verifier

MAX_CONSECUTIVE_ERRORS = 3


def run_batch(
    samples: Iterable[Sample],
    requested: int,
    main_agent: MainAgent,
    store: SupabaseStore,
    drive: GoogleDrive,
) -> dict:
    # Process one target at a time, downloading each distinct image once.
    # Checkpoint partial rounds and final previews.
    counts = Counter()
    processed = rounds = consecutive_errors = 0
    previous_file_id = None
    image = None
    print(f"Supabase run ID: {store.run_id}", flush=True)
    try:
        store.start(main_agent.settings)
        for sample in samples:
            if sample.image_file_id != previous_file_id:
                image = load_image(
                    drive.download(sample.image_file_id), sample.width, sample.height
                )
                previous_file_id = sample.image_file_id
            print(f"[{processed + 1}/{requested}] Curating ref {sample.ref_id}", flush=True)
            if image is None or image.size != (sample.width, sample.height):
                raise ValueError("Image dimensions disagree with the grounding annotation.")
            # Keep only this target's checkpoint cache, even in very large runs.
            store.saved_rounds.clear()
            completed_rounds = set()

            def checkpoint(result: Result) -> None:
                if not result.rounds:
                    return
                latest = result.rounds[-1]
                status = round_status(result, latest)
                if status == "running":
                    store.save_round(result, status)
                elif latest.number not in completed_rounds:
                    preview = render_round_preview(image, sample, latest, status)
                    buffer = BytesIO()
                    preview.save(buffer, format="PNG")
                    file_id = drive.upload(
                        PREVIEWS_FOLDER,
                        f"{store.run_id} ref {sample.ref_id} round {latest.number}.png",
                        buffer.getvalue(),
                        "image/png",
                    )
                    store.save_round(result, status, file_id)
                    completed_rounds.add(latest.number)
                    print(f"  Round {latest.number}: {status}", flush=True)
                    print(
                        f"  Round preview: https://drive.google.com/file/d/{file_id}/view",
                        flush=True,
                    )

            result = main_agent.run(sample, image, checkpoint)
            processed += 1
            rounds += len(result.rounds)
            counts[result.status] += 1
            print(f"  {result.status} after {len(result.rounds)} round(s)", flush=True)
            if result.status == "error":
                print(f"  Recorded error: {result.error}", flush=True)
                consecutive_errors += 1
                if consecutive_errors == MAX_CONSECUTIVE_ERRORS:
                    print(
                        f"  Stopping batch: {MAX_CONSECUTIVE_ERRORS} targets in a row failed. "
                        "Check the provider and rerun.",
                        flush=True,
                    )
                    break
            else:
                consecutive_errors = 0
        status = "completed" if processed == requested else "error"
        store.finish(status)
        return {
            "status": status,
            "requested": requested,
            "processed": processed,
            "unprocessed": requested - processed,
            "accepted": counts["accepted"],
            "exhausted": counts["exhausted"],
            "errors": counts["error"],
            "rounds": rounds,
            "run_id": store.run_id,
        }
    except (SupabaseError, DriveError, OSError, ValueError, KeyboardInterrupt):
        try:
            store.finish("error")
        except SupabaseError:
            print("Could not mark the run as failed in Supabase.", file=sys.stderr)
        raise


def role_models(
    settings: Settings, openai_client: ModelClient, openrouter: ModelClient
) -> dict[str, tuple[ModelClient, str]]:
    strong_clients = {"openai": openai_client, "openrouter": openrouter}
    return {
        "main_agent": (openai_client, settings.main_model),
        "challenger": (openai_client, settings.challenger_model),
        "strong_solver": (strong_clients[settings.strong_provider], settings.strong_model),
        "weak_solver": (openrouter, settings.weak_model),
        "judge": (openrouter, settings.judge_model),
    }


def positive_int(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def nonnegative_int(value: str) -> int:
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return result


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Create visual localisation training data.")
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser("drive-auth", help="Sign in to Google Drive once using a browser.")
    commands.add_parser("drive-check", help="Check Google login and root-folder access.")
    commands.add_parser("supabase-check", help="Check Supabase tables and required columns.")
    run = commands.add_parser("run", help="Curate targets read from Supabase.")
    run.add_argument("--limit", type=positive_int, default=10)
    run.add_argument(
        "--offset", type=nonnegative_int, default=0, help="Skip this many database targets."
    )
    run.add_argument(
        "--dry-run",
        action="store_true",
        help="Check cloud images/boxes without model calls or outputs.",
    )
    export = commands.add_parser(
        "export", help="Download runs' accepted examples as Qwen training data."
    )
    export.add_argument(
        "--run-id",
        type=UUID,
        required=True,
        nargs="+",
        help="One or more run IDs whose accepted rounds to export together.",
    )
    export.add_argument(
        "--output", type=Path, required=True, help="Directory for the local Qwen training bundle."
    )
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    load_dotenv(Path.cwd() / ".env", override=False)
    try:
        if args.command == "drive-auth":
            authenticate_drive()
            print(json.dumps(create_drive_client().check_connection(), indent=2))
            return 0
        if args.command == "drive-check":
            print(json.dumps(create_drive_client().check_connection(), indent=2))
            return 0
        client = create_supabase_client()
        report = check_supabase_connection(client)
        if args.command == "supabase-check":
            print(json.dumps(report, indent=2))
            if not report["ready"]:
                print("Create the schema by running the SQL files in supabase/ in numeric order.")
            return 0 if report["ready"] else 1
        if not report["ready"]:
            raise SupabaseError(
                "Supabase is not ready. Run the supabase/ SQL files in order in the SQL Editor."
            )
        drive = create_drive_client()
        drive.check_connection()
        store = SupabaseStore(client)
        if args.command == "export":
            count = export_training(
                store, drive, [str(run_id) for run_id in args.run_id], args.output
            )
            print(f"Downloaded {count} accepted examples to {args.output}")
            return 0
        settings = Settings.from_env()
        available = store.annotation_count() - args.offset
        if available < args.limit:
            raise ValueError(
                f"Only {max(0, available)} training targets available after offset {args.offset}. "
                "Import more data or reduce --limit."
            )
        samples = store.iter_samples(args.limit, offset=args.offset)
        if args.dry_run:
            for sample in samples:
                load_image(drive.download(sample.image_file_id), sample.width, sample.height)
            print(f"Validated {args.limit} cloud targets. No model calls or output files.")
            return 0
        openai_client = OpenAIClient(settings)
        openrouter = OpenRouterClient(settings)
        roles = role_models(settings, openai_client, openrouter)
        # Reject bad model IDs before run_batch creates a run row or pays for any call.
        for role, (role_client, model) in roles.items():
            role_client.check_model(model, role)
        agent = MainAgent(
            client=openai_client,
            settings=settings,
            challenger=Challenger(openai_client, settings.challenger_model),
            weak_solver=Solver(*roles["weak_solver"], role="weak_solver"),
            strong_solver=Solver(*roles["strong_solver"], role="strong_solver"),
            judge=Judge(*roles["judge"]),
            verifier=Verifier(settings),
        )
        print(f"Strong solver: {settings.strong_provider} / {settings.strong_model}")
        print(f"Weak solver: openrouter / {settings.weak_model}")
        print(f"Maximum {args.limit * settings.max_rounds * 5} model calls, excluding retries.")
        summary = run_batch(samples, args.limit, agent, store, drive)
        print(json.dumps(summary, indent=2))
        print(f"Accepted examples: rounds with status 'accepted' for run {summary['run_id']}")
        return 0 if summary["status"] == "completed" else 1
    except (
        OSError,
        ValueError,
        KeyError,
        httpx.HTTPError,
        ModelError,
        SupabaseError,
        DriveError,
    ) as error:
        # Provider wrappers intentionally avoid printing credentials or raw server bodies.
        message = (
            "Dataset request failed; check the source endpoint and connectivity."
            if isinstance(error, httpx.HTTPError)
            else str(error)
        )
        print(f"Error: {message}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
