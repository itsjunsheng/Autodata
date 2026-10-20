-- One row per round of the agent loop: the generated referring expression, both solver
-- boxes, their IoUs, and the verifier's outcome. Predictions may be null while running or
-- after a provider error. An accepted example is the accepted round plus its annotation.
-- Run after 01_refcoco_annotations.sql and 02_runs.sql.
begin;

create table public.rounds (
    run_id uuid not null references public.runs(id),
    ref_id integer not null references public.refcoco_annotations(ref_id),
    round_number integer not null check (round_number > 0),
    referring_expression text check (length(btrim(referring_expression)) between 1 and 400),
    weak_bbox_2d public.normalized_bbox,
    strong_bbox_2d public.normalized_bbox,
    weak_iou double precision check (weak_iou between 0 and 1),
    strong_iou double precision check (strong_iou between 0 and 1),
    status text not null check (status in ('running', 'accepted', 'rejected', 'error')),
    feedback text,
    preview_file_id text check (preview_file_id ~ '^[A-Za-z0-9_-]+$'),
    created_at timestamptz not null default now(),
    check ((weak_bbox_2d is null) = (weak_iou is null)),
    check ((strong_bbox_2d is null) = (strong_iou is null)),
    check (
        status not in ('accepted', 'rejected') or (
            referring_expression is not null
            and weak_bbox_2d is not null and strong_bbox_2d is not null
            and preview_file_id is not null
        )
    ),
    primary key (run_id, ref_id, round_number)
);

create unique index rounds_one_accepted_per_ref_idx
    on public.rounds(run_id, ref_id) where status = 'accepted';

-- Trusted Python backend only.
alter table public.rounds enable row level security;
revoke all on public.rounds from anon, authenticated;
grant select, insert, update on public.rounds to service_role;

notify pgrst, 'reload schema';
commit;
