create extension if not exists vector;

create table if not exists portfolio_repos (
  repo_id text primary key,
  title text not null,
  description text,
  readme text,
  url text,
  owner text,
  stars integer default 0,
  topics jsonb,
  is_showcase boolean default false,
  content_hash text,
  updated_at timestamptz default now()
);

create table if not exists portfolio_repo_chunks (
  id uuid primary key default gen_random_uuid(),
  repo_id text references portfolio_repos (repo_id) on delete cascade,
  content text not null,
  embedding vector(1536),
  content_hash text,
  chunk_index integer,
  updated_at timestamptz default now()
);

create table if not exists portfolio_profile_chunks (
  id uuid primary key default gen_random_uuid(),
  source text,
  content text not null,
  embedding vector(1536),
  chunk_index integer,
  updated_at timestamptz default now()
);

create table if not exists portfolio_contact_messages (
  id uuid primary key default gen_random_uuid(),
  name text not null,
  email text not null,
  message text not null,
  created_at timestamptz default now()
);

create or replace function match_repo_chunks(
  query_embedding vector(1536),
  match_count int
)
returns table (repo_id text, content text, score float)
language sql stable as $$
  select
    repo_id,
    content,
    1 - (embedding <=> query_embedding) as score
  from portfolio_repo_chunks
  order by embedding <=> query_embedding
  limit match_count;
$$;

create or replace function match_profile_chunks(
  query_embedding vector(1536),
  match_count int
)
returns table (content text, source text, score float)
language sql stable as $$
  select
    content,
    source,
    1 - (embedding <=> query_embedding) as score
  from portfolio_profile_chunks
  order by embedding <=> query_embedding
  limit match_count;
$$;
