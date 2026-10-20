-- One row per `run` command: the models, IoU thresholds, and execution status.
-- Run after 00_normalized_bbox.sql.
begin;

create table public.runs (
    id uuid primary key default gen_random_uuid(),
    main_model text not null,
    challenger_model text not null,
    weak_model text not null,
    strong_model text not null,
    judge_model text not null,
    iou_threshold double precision not null check (iou_threshold > 0 and iou_threshold <= 1),
    min_iou_gap double precision not null check (min_iou_gap between 0 and 1),
    max_rounds integer not null check (max_rounds between 1 and 20),
    status text not null default 'running' check (status in ('running', 'completed', 'error')),
    created_at timestamptz not null default now(),
    finished_at timestamptz
);

-- Trusted Python backend only.
alter table public.runs enable row level security;
revoke all on public.runs from anon, authenticated;
grant select, insert, update on public.runs to service_role;

notify pgrst, 'reload schema';
commit;
