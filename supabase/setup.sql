-- Fresh schema: four tables, image files in Google Drive.
begin;

-- Reuse the domain if it remains after deleting the old tables.
do $$
begin
    if not exists (
        select 1 from pg_type
        where typname = 'normalized_bbox' and typnamespace = 'public'::regnamespace
    ) then
        create domain public.normalized_bbox as double precision[]
        check (
            value is null or (
                array_ndims(value) = 1
                and array_lower(value, 1) = 1
                and array_length(value, 1) = 4
                and value[1] between 0 and 1000
                and value[2] between 0 and 1000
                and value[3] between 0 and 1000
                and value[4] between 0 and 1000
                and value[1] < value[3]
                and value[2] < value[4]
            ) is true
        );
    end if;
end;
$$;

-- One object per row; objects can share an original image.
create table public.samples (
    id uuid primary key default gen_random_uuid(),
    image_file_id text not null check (image_file_id ~ '^[A-Za-z0-9_-]+$'),
    width integer not null check (width > 0),
    height integer not null check (height > 0),
    bbox_2d public.normalized_bbox not null,
    referring_expressions text[] not null check (cardinality(referring_expressions) > 0),
    dataset text not null check (length(btrim(dataset)) > 0),
    created_at timestamptz not null default now()
);

create table public.runs (
    id uuid primary key default gen_random_uuid(),
    main_model text not null,
    challenger_model text not null,
    weak_model text not null,
    strong_model text not null,
    acceptance text not null check (acceptance in ('contrastive', 'localisation')),
    iou_threshold double precision not null check (iou_threshold > 0 and iou_threshold <= 1),
    min_iou_gap double precision not null check (min_iou_gap between 0 and 1),
    max_rounds integer not null check (max_rounds between 1 and 20),
    status text not null default 'running' check (status in ('running', 'completed', 'error')),
    created_at timestamptz not null default now(),
    finished_at timestamptz
);

-- Predictions may be null while running or after a provider error.
create table public.attempts (
    id uuid primary key default gen_random_uuid(),
    run_id uuid not null references public.runs(id),
    sample_id uuid not null references public.samples(id),
    attempt_number integer not null check (attempt_number > 0),
    referring_expression text check (length(btrim(referring_expression)) between 1 and 400),
    weak_bbox_2d public.normalized_bbox,
    strong_bbox_2d public.normalized_bbox,
    weak_iou double precision check (weak_iou between 0 and 1),
    strong_iou double precision check (strong_iou between 0 and 1),
    status text not null check (status in ('running', 'accepted', 'rejected', 'error')),
    feedback text,
    attempt_file_id text check (attempt_file_id ~ '^[A-Za-z0-9_-]+$'),
    created_at timestamptz not null default now(),
    unique (run_id, sample_id, attempt_number),
    check ((weak_bbox_2d is null) = (weak_iou is null)),
    check ((strong_bbox_2d is null) = (strong_iou is null)),
    check (
        status not in ('accepted', 'rejected') or (
            referring_expression is not null
            and weak_bbox_2d is not null and strong_bbox_2d is not null
            and attempt_file_id is not null
        )
    )
);

create index attempts_sample_id_idx on public.attempts(sample_id);
create index attempts_run_status_idx on public.attempts(run_id, status);
create index samples_dataset_id_idx on public.samples(dataset, id);
create unique index attempts_one_accepted_per_sample_idx
    on public.attempts(run_id, sample_id) where status = 'accepted';

-- Accepted training examples retain their original labels.
create table public.outputs (
    attempt_id uuid primary key references public.attempts(id),
    image_file_id text not null check (image_file_id ~ '^[A-Za-z0-9_-]+$'),
    width integer not null check (width > 0),
    height integer not null check (height > 0),
    referring_expression text not null check (length(btrim(referring_expression)) between 1 and 400),
    bbox_2d public.normalized_bbox not null,
    created_at timestamptz not null default now()
);

-- Store acceptance and its output in one transaction; retries keep the existing output.
create or replace function public.save_accepted_output()
returns trigger
language plpgsql
security invoker
set search_path = ''
as $$
begin
    insert into public.outputs (
        attempt_id, image_file_id, width, height, referring_expression, bbox_2d
    )
    select new.id, s.image_file_id, s.width, s.height, new.referring_expression, s.bbox_2d
    from public.samples s
    where s.id = new.sample_id
    on conflict (attempt_id) do nothing;
    return new;
end;
$$;

create trigger store_accepted_output
after insert or update of status on public.attempts
for each row when (new.status = 'accepted')
execute function public.save_accepted_output();

-- Trusted Python backend only.
alter table public.samples enable row level security;
alter table public.runs enable row level security;
alter table public.attempts enable row level security;
alter table public.outputs enable row level security;
revoke all on public.samples, public.runs, public.attempts, public.outputs from anon, authenticated;
grant select, insert, update on public.samples, public.runs, public.attempts, public.outputs to service_role;
revoke execute on function public.save_accepted_output() from public, anon, authenticated;
grant execute on function public.save_accepted_output() to service_role;

notify pgrst, 'reload schema';
commit;
