# Autodata visual grounding

An agentic workflow that generates referring expressions for annotated images and
evaluates them with two vision-language models. Accepted examples contain a clean
image, a referring expression, and a ground-truth bounding box for Qwen3-VL training.

The design follows the Autodata paper; see [paper notes](docs/paper-notes.md).

## Getting started

Requires Python 3.11+, OpenAI and OpenRouter API access, a Supabase project,
and a Google Drive folder.

### Python environment

Run these commands from the repository root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.lock -e .
```

Activate the environment whenever you open a new terminal. Once the prompt shows
`(.venv)`, use `python` for all subsequent commands. On macOS/Linux, activate with
`source .venv/bin/activate`.

### Configuration

Create `.env` from [.env.example](.env.example) and configure your own services:

| Variable | Value |
| --- | --- |
| `OPENAI_API_KEY` | OpenAI API key |
| `OPENROUTER_API_KEY` | OpenRouter API key |
| `SUPABASE_URL` | Your Supabase project's API URL |
| `SUPABASE_API_KEY` | Backend secret key or legacy `service_role` key |
| `GOOGLE_DRIVE_FOLDER_ID` | ID of the Drive folder used by this project |
| `GOOGLE_DRIVE_CLIENT_FILE` | OAuth client JSON; default `credentials/google_client.json` |
| `GOOGLE_DRIVE_TOKEN_FILE` | Saved login token; default `credentials/google_token.json` |

Model settings are `AUTODATA_MAIN_MODEL`, `AUTODATA_CHALLENGER_MODEL`,
`AUTODATA_STRONG_MODEL`, and `AUTODATA_WEAK_MODEL`. The OpenAI roles default to
`gpt-5.4-nano`; the Qwen weak solver defaults to `qwen/qwen3-vl-8b-instruct` on OpenRouter.
Keep `.env` and OAuth credentials private.

### Supabase

| Table | Stores |
| --- | --- |
| `samples` | Image metadata, ground-truth boxes, and source referring expressions |
| `runs` | Models, acceptance thresholds, and execution status |
| `attempts` | Generated referring expressions, predictions, IoUs, and outcomes |
| `outputs` | Accepted examples, linked to their attempt through `attempt_id` |

```powershell
python src/main.py supabase-check
```

### Google Drive

1. Enable the Google Drive API in [Google Cloud Console](https://console.cloud.google.com/).
2. Configure the OAuth consent screen and create a **Desktop app** OAuth client.
   Add your account as a test user if the app is in testing mode.
3. Save the downloaded client JSON to `credentials/google_client.json`.
4. Set `GOOGLE_DRIVE_FOLDER_ID` to your folder's ID, then sign in:

```powershell
python src/main.py drive-auth
python src/main.py drive-check
```

The sign-in creates the token file; subsequent commands refresh it automatically.
Each developer signs in with an account that can read and write to the configured
folder. See the [Google Drive Python quickstart](https://developers.google.com/workspace/drive/api/quickstart/python)
for OAuth setup details.

## Usage

### Import samples

```powershell
python src/main.py import-dataset --count 10
```

The importer selects the first 10 distinct training images from
[jxu124/refcoco](https://huggingface.co/datasets/jxu124/refcoco), with one annotated
target per image. It reads annotations from Hugging Face and downloads the original
COCO images. Images go into Drive; metadata goes into Supabase `samples`.

To expand to 100 images:

```powershell
python src/main.py import-dataset --count 100
```

`--count` is the total selection size. Repeating an import checks Supabase and Drive
before downloading, skips complete samples, and repairs missing images or metadata.
After an interrupted import, rerun the same command. Selection always follows source
order and rescans annotations from the beginning.

### Generate referring expressions

```powershell
python src/main.py run --limit 10 --max-rounds 3
```

Each target gets up to three attempts and stops when accepted. The command prints
the run ID and links to attempt images. Targets are processed in import order.
Every run starts a new experiment; it does not resume an earlier run.
Model errors stop the batch while preserving saved attempts.

| Option | Purpose |
| --- | --- |
| `--limit N` | Number of imported targets to process |
| `--max-rounds N` | Maximum attempts per target |
| `--offset N` | Skip N targets in import order |
| `--dataset NAME` | Filter targets by source dataset |
| `--iou-threshold FLOAT` | Solver success threshold; default `0.5` |
| `--min-iou-gap FLOAT` | Required strong–weak IoU gap; default `0.2` |
| `--acceptance localisation` | Accept using strong-solver success alone |
| `--dry-run` | Validate cloud inputs without model calls or output writes |

Keep the sample set unchanged when using offsets across batches. A full attempt
uses four model calls: up to 120 calls for 10 targets and three rounds, excluding retries.

### Export training data

```powershell
python src/main.py export --run-id YOUR_RUN_UUID --output training/my_experiment
```

Replace `YOUR_RUN_UUID` with a run ID. The destination must be new or empty.
Export downloads accepted clean images into `images/` and creates ShareGPT training
examples in `qwen_train.json`, with a dataset registry in `dataset_info.json`.
For LLaMA-Factory, set `dataset_dir: training/my_experiment` and
`dataset: autodata_grounding`.

## How the loop works

| Role | Responsibility |
| --- | --- |
| Main agent | Plans generation and revises instructions from previous results |
| Challenger | Generates one referring expression per attempt |
| Weak solver | Predicts a box using Qwen through OpenRouter |
| Strong solver | Predicts a box using the configured OpenAI VLM |
| Verifier | Calculates IoU against ground truth and applies acceptance rules |

The main agent and challenger see the target annotation. Both solvers receive only
the clean image and generated referring expression. Rejection feedback informs the
next attempt; accepted training labels always use the original ground-truth box.

Default **contrastive** acceptance requires all three conditions:

- Strong-solver IoU ≥ `0.5`.
- Weak-solver IoU < `0.5`.
- Strong-solver IoU − weak-solver IoU ≥ `0.2`.

The **localisation** mode checks only the strong-solver threshold, while still
running both solvers. The verifier is deterministic and does not judge expression
ambiguity or truthfulness.

Boxes use **XYXY coordinates normalised to 0–1000**; IoU scores range from **0–1**.
Training export rounds ground-truth box edges outward to integers.

## Storage

Supabase stores metadata and accepted examples. Google Drive stores images in two
top-level folders, created automatically:

```text
Drive root/
  samples/sample 1.jpg
  samples/sample 2.jpg
  attempts/<run-id>/sample 1 attempt 1.png
  attempts/<run-id>/sample 1 attempt 2.png
```

Attempt images show the referring expression, outcome, and boxes: **green** for
ground truth, **red** for the weak solver, and **blue** for the strong solver.
Training uses clean sample images. Import and generation store their results in
the cloud; `export` explicitly creates local training files.

Run one importing or generating process at a time; file creation is not coordinated
across workers. Drive access follows the folder's sharing permissions.

## Development

Agent roles each have a module in `src/`: `main_agent.py`, `challenger.py`,
`weak_solver.py`, `strong_solver.py`, and `verifier.py`.

| Supporting module | Responsibility |
| --- | --- |
| `main.py` | CLI and workflow coordination |
| `dataset.py` | RefCOCO import and image validation |
| `models.py` | Data structures, settings, and coordinate conversion |
| `model_client.py` | OpenAI requests and provider errors |
| `google_drive.py` | Google authentication and image storage |
| `supabase_client.py` | Database access and attempt persistence |
| `outputs.py` | Attempt images and training export |

```powershell
python -m pip install -e ".[dev]"
python -m pytest -q
python -m ruff check src tests
python -m ruff format --check src tests
```

Tests use mocked HTTP services and make no paid model calls. The database schema
is in [supabase/setup.sql](supabase/setup.sql).
