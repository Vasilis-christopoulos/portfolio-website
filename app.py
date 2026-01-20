"""FastAPI app exposing a LangGraph-powered agent with a GitHub showcase tool."""

import hashlib
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional, TypedDict
import math

import httpx
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field, HttpUrl
from supabase import Client, create_client

load_dotenv(Path(__file__).with_name(".env"))

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
HTTP_LOG_LEVEL = os.getenv("HTTP_LOG_LEVEL", "WARNING").upper()

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(levelname)s:%(name)s:%(message)s",
)
for noisy_logger in ("httpx", "httpcore", "hpack", "openai", "urllib3"):
    logging.getLogger(noisy_logger).setLevel(HTTP_LOG_LEVEL)

GITHUB_API_URL = "https://api.github.com"
DEFAULT_LIMIT = 20
GITHUB_USERNAME = os.getenv("GITHUB_USERNAME")
logger = logging.getLogger(__name__)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
SUPABASE_REPO_TABLE = os.getenv("SUPABASE_REPO_TABLE", "portfolio_repos")
SUPABASE_REPO_CHUNKS_TABLE = os.getenv(
    "SUPABASE_REPO_CHUNKS_TABLE", "portfolio_repo_chunks"
)
SUPABASE_PROFILE_CHUNKS_TABLE = os.getenv(
    "SUPABASE_PROFILE_CHUNKS_TABLE", "portfolio_profile_chunks"
)
SUPABASE_REPO_MATCH_RPC = os.getenv("SUPABASE_REPO_MATCH_RPC", "match_repo_chunks")
SUPABASE_PROFILE_MATCH_RPC = os.getenv(
    "SUPABASE_PROFILE_MATCH_RPC", "match_profile_chunks"
)

REPO_CACHE_TTL_SECONDS = int(os.getenv("REPO_CACHE_TTL_SECONDS", "21600"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "1536"))
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "5"))
REPO_EMBEDDINGS_ON_REFRESH = os.getenv("REPO_EMBEDDINGS_ON_REFRESH", "true").lower() in (
    "1",
    "true",
    "yes",
)
CHUNK_SIZE_CHARS = int(os.getenv("CHUNK_SIZE_CHARS", "1200"))
CHUNK_OVERLAP_CHARS = int(os.getenv("CHUNK_OVERLAP_CHARS", "200"))

SYSTEM_PROMPT = (
    "You are a portfolio data agent. If the user greets or chats, respond conversationally without tools. "
    "Use tools for explicit requests about repositories, projects, or profile/resume questions. "
    "If the request is unclear, ask a short clarification question before using tools. "
    "Choose the most appropriate tool based on its description. "
    "Use fetch_showcase_repos for listing projects. "
    "Use search_showcase_repos for targeted repo questions (e.g., 'computer vision'). "
    "Use retrieve_profile_context for resume/background questions and answer only from that context. "
    "Some tools return complete, ready-to-use data that doesn't need reformatting. "
    "For other tools, you may need to process and structure the output."
)

REPO_QUERY_KEYWORDS = (
    "repo",
    "repos",
    "repository",
    "repositories",
    "project",
    "projects",
    "showcase",
    "github",
    "readme",
    "stars",
    "topic",
    "topics",
)

PROFILE_QUERY_KEYWORDS = (
    "resume",
    "cv",
    "background",
    "experience",
    "skills",
    "hire",
    "hiring",
    "why should i hire",
    "profile",
    "bio",
    "about you",
)


def wants_repo_data(text: str) -> bool:
    if not text:
        return False
    lowered = text.lower()
    return any(keyword in lowered for keyword in REPO_QUERY_KEYWORDS)


def wants_profile_data(text: str) -> bool:
    if not text:
        return False
    lowered = text.lower()
    return any(keyword in lowered for keyword in PROFILE_QUERY_KEYWORDS)


def wants_tool_data(text: str) -> bool:
    return wants_repo_data(text) or wants_profile_data(text)


class AgentState(TypedDict):
    messages: Annotated[List[Any], add_messages]
    repos: Optional[List[Dict[str, Any]]]


class AgentRequest(BaseModel):
    query: str = Field(default="Fetch showcase repositories for the portfolio UI.")
    limit: int = Field(default=DEFAULT_LIMIT, ge=1, le=50)
    session_id: Optional[str] = Field(
        default=None,
        description="Client-provided session/thread id for conversation memory.",
    )
    use_agent: bool = Field(
        default=True,
        description="Use the LLM agent. Set false to call the GitHub tool directly (useful for tests).",
    )


class ShowcaseRepo(BaseModel):
    repo_id: Optional[str] = None
    title: str
    description: Optional[str] = None
    readme: str = ""
    url: HttpUrl
    owner: str
    stars: int = 0
    topics: List[str] = Field(default_factory=list)


class AgentResponse(BaseModel):
    repos: List[ShowcaseRepo]
    source: str
    raw_output: Optional[Any] = None


_supabase_client: Optional[Client] = None
_embeddings_client: Optional[OpenAIEmbeddings] = None


def get_supabase() -> Client:
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_KEY are required.")
    global _supabase_client
    if _supabase_client is None:
        _supabase_client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    return _supabase_client


def get_embeddings() -> OpenAIEmbeddings:
    global _embeddings_client
    if _embeddings_client is None:
        _embeddings_client = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            dimensions=EMBEDDING_DIMENSION,
        )
    return _embeddings_client


def supabase_response_data(response: Any, action: str) -> List[Dict[str, Any]]:
    error = getattr(response, "error", None)
    if error:
        logger.error("Supabase error on %s: %s", action, error)
        raise RuntimeError(f"Supabase {action} failed: {error}")
    return response.data or []


def to_pgvector_literal(vector: List[float]) -> str:
    parts = []
    for value in vector:
        if not math.isfinite(value):
            raise ValueError("Embedding vector contains non-finite values.")
        parts.append(f"{value:.10f}")
    return "[" + ",".join(parts) + "]"


def parse_timestamp(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        if value.endswith("Z"):
            value = value.replace("Z", "+00:00")
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def chunk_text(text: str) -> List[str]:
    cleaned = text.strip()
    if not cleaned:
        return []
    chunks: List[str] = []
    start = 0
    length = len(cleaned)
    while start < length:
        end = min(length, start + CHUNK_SIZE_CHARS)
        chunk = cleaned[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= length:
            break
        start = max(end - CHUNK_OVERLAP_CHARS, start + 1)
    return chunks


def repo_content_hash(repo: Dict[str, Any]) -> str:
    raw = "\n".join(
        [
            str(repo.get("title") or ""),
            str(repo.get("description") or ""),
            str(repo.get("readme") or ""),
        ]
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def build_repo_embedding_text(repo: Dict[str, Any]) -> str:
    parts = []
    title = repo.get("title")
    description = repo.get("description")
    readme = repo.get("readme")
    if title:
        parts.append(f"Title: {title}")
    if description:
        parts.append(f"Description: {description}")
    if readme:
        parts.append(f"README: {readme}")
    return "\n".join(parts)


def fetch_cached_showcase_repo_ids(limit: int) -> Optional[List[str]]:
    client = get_supabase()
    response = (
        client.table(SUPABASE_REPO_TABLE)
        .select("repo_id, updated_at, stars")
        .eq("is_showcase", True)
        .order("stars", desc=True)
        .limit(limit)
        .execute()
    )
    rows = supabase_response_data(response, "fetch_cached_showcase_repo_ids")
    if not rows:
        return None
    cutoff = datetime.now(timezone.utc) - timedelta(seconds=REPO_CACHE_TTL_SECONDS)
    for row in rows:
        updated_at = parse_timestamp(row.get("updated_at"))
        if updated_at is None or updated_at < cutoff:
            return None
    return [row["repo_id"] for row in rows if row.get("repo_id")]


def load_repos_by_ids(repo_ids: List[str]) -> List[Dict[str, Any]]:
    if not repo_ids:
        return []
    client = get_supabase()
    response = (
        client.table(SUPABASE_REPO_TABLE)
        .select("*")
        .in_("repo_id", repo_ids)
        .execute()
    )
    rows = supabase_response_data(response, "load_repos_by_ids")
    rows_by_id = {row.get("repo_id"): row for row in rows}
    ordered = [rows_by_id[repo_id] for repo_id in repo_ids if repo_id in rows_by_id]
    return ordered


def upsert_repos_to_cache(repos: List[Dict[str, Any]]) -> None:
    if not repos:
        return
    client = get_supabase()
    now = datetime.now(timezone.utc).isoformat()
    payload = []
    for repo in repos:
        payload.append(
            {
                "repo_id": repo.get("repo_id"),
                "title": repo.get("title"),
                "description": repo.get("description"),
                "readme": repo.get("readme"),
                "url": repo.get("url"),
                "owner": repo.get("owner"),
                "stars": repo.get("stars", 0),
                "topics": repo.get("topics", []),
                "is_showcase": repo.get("is_showcase", True),
                "content_hash": repo.get("content_hash"),
                "updated_at": now,
            }
        )
    response = client.table(SUPABASE_REPO_TABLE).upsert(payload).execute()
    supabase_response_data(response, "upsert_repos_to_cache")


def get_repo_hashes(repo_ids: List[str]) -> Dict[str, str]:
    if not repo_ids:
        return {}
    client = get_supabase()
    response = (
        client.table(SUPABASE_REPO_TABLE)
        .select("repo_id, content_hash")
        .in_("repo_id", repo_ids)
        .execute()
    )
    rows = supabase_response_data(response, "get_repo_hashes")
    return {row.get("repo_id"): row.get("content_hash") for row in rows}


def refresh_repo_chunks(
    repos: List[Dict[str, Any]],
    existing_hashes: Optional[Dict[str, str]] = None,
) -> None:
    if not repos or not REPO_EMBEDDINGS_ON_REFRESH:
        return
    client = get_supabase()
    embeddings = get_embeddings()
    if existing_hashes is None:
        existing_hashes = get_repo_hashes([repo["repo_id"] for repo in repos])
    for repo in repos:
        repo_id = repo["repo_id"]
        content_hash = repo.get("content_hash")
        if not repo_id or not content_hash:
            continue
        if existing_hashes.get(repo_id) == content_hash:
            continue
        delete_response = (
            client.table(SUPABASE_REPO_CHUNKS_TABLE)
            .delete()
            .eq("repo_id", repo_id)
            .execute()
        )
        supabase_response_data(delete_response, "delete_repo_chunks")
        chunks = chunk_text(build_repo_embedding_text(repo))
        if not chunks:
            continue
        vectors = embeddings.embed_documents(chunks)
        payload = []
        for idx, (chunk, vector) in enumerate(zip(chunks, vectors)):
            payload.append(
                {
                    "repo_id": repo_id,
                    "content": chunk,
                    "embedding": vector,
                    "content_hash": content_hash,
                    "chunk_index": idx,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
            )
        insert_response = (
            client.table(SUPABASE_REPO_CHUNKS_TABLE).insert(payload).execute()
        )
        supabase_response_data(insert_response, "insert_repo_chunks")


def match_repo_chunks(query: str, limit: int) -> List[Dict[str, Any]]:
    client = get_supabase()
    embeddings = get_embeddings()
    query_vector = to_pgvector_literal(embeddings.embed_query(query))
    response = client.rpc(
        SUPABASE_REPO_MATCH_RPC,
        {
            "query_embedding": query_vector,
            "match_count": limit,
        },
    ).execute()
    return supabase_response_data(response, "match_repo_chunks")


def match_profile_chunks(query: str, limit: int) -> List[Dict[str, Any]]:
    client = get_supabase()
    embeddings = get_embeddings()
    query_vector = to_pgvector_literal(embeddings.embed_query(query))
    response = client.rpc(
        SUPABASE_PROFILE_MATCH_RPC,
        {
            "query_embedding": query_vector,
            "match_count": limit,
        },
    ).execute()
    return supabase_response_data(response, "match_profile_chunks")


def get_showcase_repos(limit: int) -> List[Dict[str, Any]]:
    limit = max(1, min(limit, 50))
    repo_ids = fetch_cached_showcase_repo_ids(limit)
    if not repo_ids:
        repos = _fetch_showcase_repos(limit)
        existing_hashes = get_repo_hashes([repo["repo_id"] for repo in repos])
        upsert_repos_to_cache(repos)
        refresh_repo_chunks(repos, existing_hashes)
        repo_ids = [repo.get("repo_id") for repo in repos if repo.get("repo_id")]
    return load_repos_by_ids(repo_ids)

def github_headers() -> Dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "portfolio-agent",
    }
    token = os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _fetch_showcase_repos(limit: int) -> List[Dict[str, Any]]:
    if not GITHUB_USERNAME:
        raise RuntimeError("GITHUB_USERNAME is required to scope results to your repos.")
    limit = max(1, min(limit, 50))
    repos: List[Dict[str, Any]] = []
    try:
        with httpx.Client(timeout=15.0, headers=github_headers()) as client:
            search_resp = client.get(
                f"{GITHUB_API_URL}/search/repositories",
                params={
                    "q": f"user:{GITHUB_USERNAME} topic:showcase",
                    "sort": "stars",
                    "order": "desc",
                    "per_page": limit,
                },
            )
            search_resp.raise_for_status()
            for item in search_resp.json().get("items", []):
                owner = item["owner"]["login"]
                name = item["name"]
                readme_resp = client.get(
                    f"{GITHUB_API_URL}/repos/{owner}/{name}/readme",
                    headers={
                        "Accept": "application/vnd.github.raw",
                        "User-Agent": "portfolio-agent",
                    },
                )
                readme_text = readme_resp.text if readme_resp.status_code == 200 else ""
                repo_id = item.get("full_name", f"{owner}/{name}")
                repo_data = ShowcaseRepo(
                    repo_id=repo_id,
                    title=item.get("full_name", name),
                    description=item.get("description"),
                    readme=readme_text,
                    url=item.get("html_url"),
                    owner=owner,
                    stars=item.get("stargazers_count", 0),
                    topics=item.get("topics", []),
                ).model_dump(mode="json")
                repo_data["is_showcase"] = True
                repo_data["content_hash"] = repo_content_hash(repo_data)
                repos.append(repo_data)
        return repos
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(
            f"GitHub API error {exc.response.status_code}: {exc.response.text}"
        ) from exc
    except httpx.RequestError as exc:
        raise RuntimeError(f"GitHub request failed: {exc}") from exc


@tool
def fetch_showcase_repos(limit: int = DEFAULT_LIMIT) -> Dict[str, Any]:
    """Fetch showcase repos and cache them. Returns repo ids for lookup."""
    limit = max(1, min(limit, 50))
    cached_repo_ids = fetch_cached_showcase_repo_ids(limit)
    if cached_repo_ids:
        logger.info(
            "fetch_showcase_repos: source=cache repos=%d",
            len(cached_repo_ids),
        )
        return {"repo_ids": cached_repo_ids, "source": "cache"}
    repos = _fetch_showcase_repos(limit)
    existing_hashes = get_repo_hashes([repo["repo_id"] for repo in repos])
    upsert_repos_to_cache(repos)
    refresh_repo_chunks(repos, existing_hashes)
    repo_ids = [repo.get("repo_id") for repo in repos if repo.get("repo_id")]
    logger.info("fetch_showcase_repos: source=github repos=%d", len(repo_ids))
    return {"repo_ids": repo_ids, "source": "github"}


@tool
def search_showcase_repos(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Search cached repo content for targeted questions."""
    limit = max(1, min(limit, 10))
    try:
        matches = match_repo_chunks(query, limit)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Repo chunk match failed", exc_info=exc)
        matches = []
    if not matches:
        try:
            client = get_supabase()
            response = (
                client.table(SUPABASE_REPO_TABLE)
                .select("repo_id, title, description, url, owner, stars, topics")
                .or_(f"title.ilike.%{query}%,description.ilike.%{query}%")
                .limit(limit)
                .execute()
            )
            fallback_rows = supabase_response_data(response, "fallback_repo_search")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Repo fallback search failed", exc_info=exc)
            fallback_rows = []
        results = [
            {
                "repo_id": row.get("repo_id"),
                "title": row.get("title"),
                "description": row.get("description"),
                "url": row.get("url"),
                "owner": row.get("owner"),
                "stars": row.get("stars"),
                "topics": row.get("topics", []),
                "snippets": [],
                "score": None,
            }
            for row in fallback_rows
        ]
        logger.info(
            "search_showcase_repos: query=%r hits=%d fallback=true",
            query,
            len(results),
        )
        return results
    repo_hits: Dict[str, Dict[str, Any]] = {}
    for match in matches:
        repo_id = match.get("repo_id")
        if not repo_id:
            continue
        hit = repo_hits.setdefault(
            repo_id,
            {
                "repo_id": repo_id,
                "score": match.get("score"),
                "snippets": [],
            },
        )
        snippet = match.get("content")
        if snippet:
            hit["snippets"].append(snippet)
    repo_ids = list(repo_hits.keys())
    repo_rows = load_repos_by_ids(repo_ids)
    rows_by_id = {row.get("repo_id"): row for row in repo_rows}
    results = []
    for repo_id in repo_ids:
        row = rows_by_id.get(repo_id, {})
        hit = repo_hits.get(repo_id, {})
        results.append(
            {
                "repo_id": repo_id,
                "title": row.get("title"),
                "description": row.get("description"),
                "url": row.get("url"),
                "owner": row.get("owner"),
                "stars": row.get("stars"),
                "topics": row.get("topics", []),
                "snippets": hit.get("snippets", []),
                "score": hit.get("score"),
            }
        )
    logger.info(
        "search_showcase_repos: query=%r hits=%d fallback=false",
        query,
        len(results),
    )
    return results


@tool
def retrieve_profile_context(query: str, limit: int = RAG_TOP_K) -> List[Dict[str, Any]]:
    """Retrieve profile context chunks for answering resume/background questions."""
    limit = max(1, min(limit, 10))
    matches = match_profile_chunks(query, limit)
    results = []
    for match in matches:
        results.append(
            {
                "content": match.get("content"),
                "source": match.get("source"),
                "score": match.get("score"),
            }
        )
    logger.info("retrieve_profile_context: query=%r hits=%d", query, len(results))
    return results

# Registry of tools that return ready-to-use data and don't need LLM post-processing
DIRECT_RETURN_TOOLS = {"fetch_showcase_repos"}
memory_saver = MemorySaver()


def build_agent_graph():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required to run the agent.")

    tools = [fetch_showcase_repos, search_showcase_repos, retrieve_profile_context]
    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0,
    )
    llm_with_tools = llm.bind_tools(tools)

    async def call_model(state: AgentState) -> Dict[str, Any]:
        messages = state.get("messages", [])
        last_user_text = ""
        for message in reversed(messages):
            if isinstance(message, HumanMessage):
                last_user_text = message.content
                break
        use_tools = wants_tool_data(last_user_text)
        if not use_tools:
            recent_messages = messages[-4:] if len(messages) >= 4 else messages
            for message in reversed(recent_messages):
                if isinstance(message, HumanMessage) and wants_tool_data(message.content):
                    use_tools = True
                    break
        model = llm_with_tools if use_tools else llm
        response = await model.ainvoke(
            [SystemMessage(content=SYSTEM_PROMPT), *messages]
        )
        return {"messages": [response]}

    def should_skip_llm(state: AgentState) -> bool:
        """Check if the last tool call was a direct_return tool."""
        messages = state.get("messages", [])
        if not messages:
            return False
        
        # Find the most recent AIMessage with tool_calls to see which tool was called
        for msg in reversed(messages):
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                tool_name = msg.tool_calls[0]['name']
                return tool_name in DIRECT_RETURN_TOOLS
        
        return False
    
    def route_agent(state: AgentState) -> str:
        messages = state.get("messages", [])
        if not messages:
            return "end"
        last_message = messages[-1]
        if getattr(last_message, "tool_calls", None):
            return "tools"
        return "end"
    
    def route_after_tools(state: AgentState) -> str:
        """Route after tool execution: skip LLM for direct_return tools."""
        if should_skip_llm(state):
            return "end"
        return "agent"

    graph = StateGraph(AgentState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        route_agent,
        {
            "tools": "tools",
            "end": END,
        },
    )
    # After tools, conditionally route: direct_return tools skip LLM
    graph.add_conditional_edges(
        "tools",
        route_after_tools,
        {
            "agent": "agent",
            "end": END,
        },
    )
    return graph.compile(checkpointer=memory_saver)


agent_graph = None


def get_agent_graph():
    global agent_graph
    if agent_graph is None:
        agent_graph = build_agent_graph()
    return agent_graph


def extract_repos_from_messages(messages: List[Any]) -> List[Dict[str, Any]]:
    logger.debug(f"extract_repos_from_messages called with {len(messages)} messages")
    # Prefer tool message output (raw tool result).
    for message in reversed(messages):
        logger.debug(f"Processing message: {type(message).__name__}")
        if isinstance(message, ToolMessage):
            content = message.content
            logger.debug(f"ToolMessage content type: {type(content)}")
            if isinstance(content, list):
                logger.debug(f"Returning list directly with {len(content)} items")
                return content
            if isinstance(content, str):
                logger.debug(f"ToolMessage content (first 100 chars): {content[:100]}")
                # Try JSON parsing first
                try:
                    parsed = json.loads(content)
                    if isinstance(parsed, dict) and "repos" in parsed:
                        logger.debug("Found repos in JSON dict")
                        return parsed["repos"]
                    if isinstance(parsed, list):
                        logger.debug(f"Found JSON list with {len(parsed)} items")
                        return parsed
                except json.JSONDecodeError as e:
                    logger.debug(f"JSON decode failed: {e}")
                    pass
                # Try Python literal eval for strings like "[{'key': 'value'}]"
                try:
                    import ast
                    parsed = ast.literal_eval(content)
                    if isinstance(parsed, list):
                        logger.debug(f"Found Python list with {len(parsed)} items via literal_eval")
                        return parsed
                    if isinstance(parsed, dict) and "repos" in parsed:
                        logger.debug("Found repos in Python dict via literal_eval")
                        return parsed["repos"]
                except (ValueError, SyntaxError) as e:
                    logger.debug(f"literal_eval failed: {e}")
                    continue
    # Fallback: look at last message content for JSON.
    for message in reversed(messages):
        content = getattr(message, "content", None)
        if isinstance(content, str):
            try:
                parsed = json.loads(content)
                if isinstance(parsed, dict) and "repos" in parsed:
                    return parsed["repos"]
            except json.JSONDecodeError:
                continue
    return []


def extract_repo_ids_from_messages(messages: List[Any]) -> List[str]:
    logger.debug(f"extract_repo_ids_from_messages called with {len(messages)} messages")
    repo_ids: List[str] = []
    for message in reversed(messages):
        if not isinstance(message, ToolMessage):
            continue
        content = message.content
        if isinstance(content, dict):
            if "repo_ids" in content and isinstance(content["repo_ids"], list):
                repo_ids.extend([repo_id for repo_id in content["repo_ids"] if repo_id])
            if "repo_id" in content and content["repo_id"]:
                repo_ids.append(content["repo_id"])
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("repo_id"):
                    repo_ids.append(item["repo_id"])
        elif isinstance(content, str):
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                parsed = None
            if parsed is None:
                try:
                    import ast

                    parsed = ast.literal_eval(content)
                except (ValueError, SyntaxError):
                    parsed = None
            if isinstance(parsed, dict):
                if isinstance(parsed.get("repo_ids"), list):
                    repo_ids.extend([repo_id for repo_id in parsed["repo_ids"] if repo_id])
                if parsed.get("repo_id"):
                    repo_ids.append(parsed["repo_id"])
            elif isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict) and item.get("repo_id"):
                        repo_ids.append(item["repo_id"])
    deduped: List[str] = []
    seen = set()
    for repo_id in repo_ids:
        if repo_id in seen:
            continue
        seen.add(repo_id)
        deduped.append(repo_id)
    return deduped


app = FastAPI(title="Portfolio Agent API", version="0.1.0")

# Allow browser calls; replace origins with specific UI domains when known.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/agent/showcase", response_model=AgentResponse)
async def agent_showcase(request: AgentRequest) -> AgentResponse:
    if request.use_agent:
        try:
            agent = get_agent_graph()
            agent_input = request.query
            if wants_repo_data(request.query):
                agent_input = (
                    f"{request.query}\n"
                    f"If you call fetch_showcase_repos, use limit {request.limit}. "
                    "If you call search_showcase_repos, use limit 5."
                )
            logger.info("Agent invocation", extra={"limit": request.limit, "query": request.query})
            thread_id = request.session_id or "default"
            result = await agent.ainvoke(
                {"messages": [HumanMessage(content=agent_input)]},
                config={"configurable": {"thread_id": thread_id}},
            )
            logger.debug(f"Agent result: {result}")
            messages = result.get("messages", [])
            logger.debug(f"Messages: {messages}")
            turn_messages = messages
            for i in range(len(messages) - 1, -1, -1):
                if isinstance(messages[i], HumanMessage):
                    turn_messages = messages[i + 1 :]
                    break
            tool_called = any(isinstance(message, ToolMessage) for message in turn_messages)
            repos: List[Dict[str, Any]] = []
            if tool_called:
                repo_ids = extract_repo_ids_from_messages(turn_messages)
                if repo_ids:
                    repos = load_repos_by_ids(repo_ids)
                if wants_repo_data(request.query) and not repos:
                    repos = extract_repos_from_messages(turn_messages)
                logger.debug(f"Extracted repos: {repos}")
                if wants_repo_data(request.query) and not repos:
                    raise HTTPException(
                        status_code=502,
                        detail="Agent did not return repository data.",
                    )
            final_text = ""
            if messages:
                final_text = getattr(messages[-1], "content", "")
            return AgentResponse(
                repos=[ShowcaseRepo(**repo) for repo in repos] if repos else [],
                source="agent",
                raw_output=final_text,
            )
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("Agent failed")
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    try:
        repos = get_showcase_repos(request.limit)
        return AgentResponse(
            repos=[ShowcaseRepo(**repo) for repo in repos],
            source="tool",
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=str(exc)) from exc


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
