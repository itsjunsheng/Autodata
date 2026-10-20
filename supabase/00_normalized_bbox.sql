-- Shared coordinate type: XYXY box edges normalised to 0-1000, x1 < x2 and y1 < y2.
-- Run first; every table file depends on it.
begin;

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

commit;
