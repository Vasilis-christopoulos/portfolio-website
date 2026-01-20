# portfolio-website
Personal AI powered portfolio website. Chatbot interface tailored to recruiters and hiring managers, answering questions based on profile and work.

## Supabase cache + RAG setup
1) Create the schema in `supabase_schema.sql`.
2) Set environment variables:
   - `SUPABASE_URL`
   - `SUPABASE_SERVICE_KEY`
   - `OPENAI_API_KEY`
   - `GITHUB_USERNAME`
   - `GITHUB_TOKEN` (optional)
   - `EMBEDDING_MODEL` (default `text-embedding-3-small`)
   - `EMBEDDING_DIMENSION` (default `1536`, must match schema)
   - `REPO_CACHE_TTL_SECONDS` (default `21600`)
   - `REPO_EMBEDDINGS_ON_REFRESH` (default `true`, set `false` to index offline)
3) Index profile docs for RAG:
   - `python scripts/index_profile_docs.py path/to/docs`

## Run
`uvicorn app:app --reload`
