-- Run this in Supabase SQL Editor or via supabase db push

create extension if not exists vector;

create table garments (
    id                  uuid primary key default gen_random_uuid(),
    user_id             uuid not null references auth.users(id) on delete cascade,
    category            text not null,
    subcategory         text,
    colors              jsonb not null default '[]',
    pattern             text,
    fabric              text,
    brand               text,
    season              text[] default '{}',
    occasions           text[] default '{}',
    care_instructions   text,
    style_tags          text[] default '{}',
    formality_score     smallint check (formality_score between 0 and 10),
    original_image_url  text,
    segmented_image_url text,
    thumbnail_url       text,
    embedding           vector(512),
    sync_status         text not null default 'PENDING'
                          check (sync_status in ('PENDING','PROCESSING','SYNCED','FAILED')),
    worn_count          integer not null default 0,
    last_worn_date      date,
    is_favorite         boolean not null default false,
    notes               text,
    created_at          timestamptz not null default now(),
    updated_at          timestamptz not null default now()
);

create index garments_embedding_hnsw on garments
    using hnsw (embedding vector_cosine_ops) with (m=16, ef_construction=128);
create index garments_user_category on garments (user_id, category);
create index garments_user_created  on garments (user_id, created_at desc);
create index garments_user_sync     on garments (user_id, sync_status) where sync_status != 'SYNCED';

alter table garments enable row level security;
create policy "own_garments" on garments
    using (auth.uid() = user_id) with check (auth.uid() = user_id);

create table outfits (
    id              uuid primary key default gen_random_uuid(),
    user_id         uuid not null references auth.users(id) on delete cascade,
    name            text not null,
    garment_ids     uuid[] not null default '{}',
    occasion        text,
    season          text,
    thumbnail_url   text,
    worn_count      integer not null default 0,
    last_worn_date  date,
    created_at      timestamptz not null default now()
);

alter table outfits enable row level security;
create policy "own_outfits" on outfits
    using (auth.uid() = user_id) with check (auth.uid() = user_id);

create or replace function similar_garments(target_id uuid, result_limit int default 10)
returns table (id uuid, similarity float) language sql security definer as $$
    select g.id, 1 - (g.embedding <=> t.embedding) as similarity
    from garments g join garments t on t.id = target_id
    where g.user_id = t.user_id and g.id != target_id
    order by g.embedding <=> t.embedding limit result_limit;
$$;
