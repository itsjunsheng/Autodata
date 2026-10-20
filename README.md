# Autodata visual grounding

An agentic workflow that generates referring expressions for annotated images and
evaluates them with two vision-language models. Accepted examples contain a clean
image, a referring expression, and a ground-truth bounding box for Qwen3-VL training.

The design follows the Autodata paper; see [paper notes](docs/paper-notes.md).

Three commands make up the workflow. `python src/dataset.py` imports RefCOCO images into
Drive and their annotations into Supabase. `python src/main.py run` generates and tests
referring expressions for those annotations. `python src/main.py export` writes the
accepted ones as a Qwen training set.

## Getting started

Requires Python 3.11+, OpenAI and OpenRouter API access, a Supabase project,
and a Google Drive folder with two subfolders, `images` and `previews`.

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

The five model settings are required and have no defaults in code. `AUTODATA_MAIN_MODEL`
and `AUTODATA_CHALLENGER_MODEL` name OpenAI models for the main agent and the challenger.
`AUTODATA_STRONG_MODEL` names the strong solver, an OpenRouter model unless
`AUTODATA_STRONG_PROVIDER=openai` is set. `AUTODATA_WEAK_MODEL` names the Qwen weak
solver on OpenRouter, the checkpoint you intend to fine-tune. `AUTODATA_JUDGE_MODEL` names the OpenRouter
vision model that reviews passing expressions for truth and uniqueness; use a family other
than the challenger's. `run` stops with a message
naming any missing variable, and before any paid call it checks that each provider
offers the configured model ID. Main-agent and challenger calls run at low reasoning
effort with an 8,000-token output cap; solver calls use greedy decoding.
Keep `.env` and OAuth credentials private.

### Supabase

| Table | Stores |
| --- | --- |
| `refcoco_annotations` | One row per annotated object: `ref_id` (primary key), COCO image ID, category, image size, ground-truth box, the human referring expressions, and the Drive file ID of its image |
| `runs` | One row per `run` command: models, IoU thresholds, and status |
| `rounds` | One row per loop round, keyed by run, `ref_id`, and round number: the generated referring expression, both solver boxes, IoUs, and the outcome. Rows with status `accepted` are the training examples |

The schema is one SQL file per table in [supabase/](supabase/), plus the shared coordinate
type. In the Supabase SQL editor run them in numeric order, `00_normalized_bbox.sql` through
`03_rounds.sql`, on an empty project. Then check the result:

```powershell
python src/main.py supabase-check
```

### Google Drive

1. Enable the Google Drive API in [Google Cloud Console](https://console.cloud.google.com/).
2. Configure the OAuth consent screen and create a **Desktop app** OAuth client.
   Add your account as a test user if the app is in testing mode.
3. Save the downloaded client JSON to `credentials/google_client.json`.
4. Create two folders named `images` and `previews` inside the project folder. The
   code never creates folders; it only looks these two up.
5. Set `GOOGLE_DRIVE_FOLDER_ID` to the project folder's ID, then sign in and check:

```powershell
python src/main.py drive-auth
python src/main.py drive-check
```

The sign-in creates the token file; subsequent commands refresh it automatically.
Each developer signs in with an account that can read and write to the configured
folder. See the [Google Drive Python quickstart](https://developers.google.com/workspace/drive/api/quickstart/python)
for OAuth setup details.

## Usage

### Import annotations

```powershell
python src/dataset.py --count 10
```

The importer takes the first 10 images of the
[jxu124/refcoco](https://huggingface.co/datasets/jxu124/refcoco) training split in source
order and imports every annotated object in them, each with its human referring
expressions. It reads annotations from Hugging Face and downloads the original COCO
images. Each image goes into the Drive `images/` folder under its COCO image ID; each
object becomes one row in `refcoco_annotations` pointing at that file.

To expand to 100 images:

```powershell
python src/dataset.py --count 100
```

`--count` is the number of images; each contributes all of its annotated objects, about
2.5 on average. There is no offset: selection always starts from the first image, so a
larger count adds to what is already imported. Rerunning the same command skips objects
that are already complete, repairs a missing image or row, and never duplicates either.
After an interrupted import, rerun the same command.

### Generate referring expressions

```powershell
python src/main.py run --limit 10
```

Each target gets up to six rounds and stops when accepted. The command prints
the run ID and links to round previews. Targets are processed in import order.
Every run starts a new experiment; it does not resume an earlier run.
A model error marks that target's round `error` with the message in `feedback`, and
the batch continues with the next target. After three consecutive failed targets the
batch stops and the run status is `error`.

| Option | Purpose |
| --- | --- |
| `--limit N` | Number of imported targets to process |
| `--offset N` | Skip N targets in import order |
| `--dry-run` | Validate cloud inputs without model calls or output writes |

Keep the imported set unchanged when using offsets across batches. A full round
uses up to five model calls, the fifth only when the three checks pass: up to 300 calls for
10 targets and six rounds, excluding retries.

### Export training data

```powershell
python src/main.py export --run-id YOUR_RUN_UUID --output training/my_experiment
```

Replace `YOUR_RUN_UUID` with one or more run IDs; accepted rounds from all of them go
into one bundle. The destination must be new or empty.
Export reads the accepted rounds joined to their annotations, downloads each
referenced original image once into `images/` under the output directory, and writes
ShareGPT training examples to `qwen_train.json`, with a dataset registry in
`dataset_info.json`.
For LLaMA-Factory, set `dataset_dir: training/my_experiment` and
`dataset: autodata_grounding`.

## How the loop works

| Role | Responsibility |
| --- | --- |
| Main agent | Plans generation and revises instructions from previous results |
| Challenger | Generates one referring expression per round |
| Weak solver | Predicts a box using Qwen through OpenRouter |
| Strong solver | Predicts a box using Qwen3-VL-235B on OpenRouter by default, or an OpenAI VLM |
| Verifier | Calculates IoU against ground truth and applies the three accept checks |
| Judge | Sees the marked target; rejects a passing expression whose claims are not visible or that fits another object |

The main agent and challenger see the target annotation. Both solvers receive only
the clean image and generated referring expression. Rejection feedback informs the
next round; accepted training labels always use the original ground-truth box.
Both solvers receive exactly the user prompt written into the training file, with no
system turn and greedy decoding, so the weak solver is measured on the input it will be
trained and evaluated on. That prompt is `grounding_prompt` in `solver.py`; use it for
evaluation too.

A round is accepted only when all three conditions hold:

- Strong-solver IoU ≥ `0.5`.
- Weak-solver IoU < `0.5`.
- Strong-solver IoU − weak-solver IoU ≥ `0.2`.

The threshold, the gap and the round limit are defaults on `Settings` in `models.py`.

The verifier is deterministic. A round that passes it then goes to the judge, a model call
that checks every stated claim is visible on the target and that no other object fits; its
rejections keep status `rejected` with feedback starting `Rejected by judge`.

Boxes use **XYXY coordinates normalised to 0–1000**; IoU scores range from **0–1**.
Training export rounds ground-truth box edges outward to integers.

## Storage

Supabase stores annotations, runs and rounds. Google Drive stores images in the two
folders you created inside the project folder:

```text
Drive root/
  images/581857.jpg
  previews/<run id> ref 0 round 1.png
```

`images/` holds one original per COCO image, named by its COCO image ID. `previews/`
holds one PNG per loop round, named by run ID, RefCOCO ref ID and round number.

Round previews show the referring expression, outcome, and boxes: **green** for
ground truth, **red** for the weak solver, and **blue** for the strong solver.
Training uses the clean original images. Import and generation store their results in
the cloud; `export` explicitly creates local training files.

Run one importing or generating process at a time; file creation is not coordinated
across workers. Drive access follows the folder's sharing permissions.

## Development

Agent roles each have a module in `src/`: `main_agent.py`, `challenger.py`,
`solver.py` (one class, used for both the weak and the strong solver), `judge.py`, and
`verifier.py`.

| Supporting module | Responsibility |
| --- | --- |
| `main.py` | CLI for Drive and Supabase checks, the agent loop, and export |
| `dataset.py` | RefCOCO import script (`python src/dataset.py`), category lookup, and image validation |
| `models.py` | Data structures, settings, and coordinate conversion |
| `model_client.py` | OpenAI and OpenRouter requests, model pre-flight checks, and provider errors |
| `google_drive.py` | Google authentication and image storage |
| `supabase_client.py` | Database access and round persistence |
| `outputs.py` | Round previews and training export |

```powershell
python -m pip install -e ".[dev]"
python -m pytest -q
python -m ruff check src tests
python -m ruff format --check src tests
```

Tests use mocked HTTP services and make no paid model calls. The database schema
is in [supabase/](supabase/), one file per table plus the shared box type.
