-- One RefCOCO object per row; objects can share an original image stored in Drive.
-- Run after 00_normalized_bbox.sql.
begin;

create table public.refcoco_annotations (
    ref_id integer primary key check (ref_id >= 0),
    image_file_id text not null check (image_file_id ~ '^[A-Za-z0-9_-]+$'),
    width integer not null check (width > 0),
    height integer not null check (height > 0),
    bbox_2d public.normalized_bbox not null,
    referring_expressions text[] not null check (cardinality(referring_expressions) > 0),
    coco_image_id integer not null check (coco_image_id > 0),
    category text not null check (length(btrim(category)) > 0),
    created_at timestamptz not null default now()
);

-- Trusted Python backend only.
alter table public.refcoco_annotations enable row level security;
revoke all on public.refcoco_annotations from anon, authenticated;
grant select, insert, update on public.refcoco_annotations to service_role;

notify pgrst, 'reload schema';
commit;
