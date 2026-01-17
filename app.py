"""FastAPI app exposing a LangGraph-powered agent with a GitHub showcase tool."""

import json
import os
from typing import Any, Dict, List, Optional

import httpx
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field, HttpUrl

load_dotenv()

GITHUB_API_URL = "https://api.github.com"
DEFAULT_LIMIT = 20
GITHUB_USERNAME = os.getenv("GITHUB_USERNAME")


class AgentRequest(BaseModel):
    query: str = Field(default="Fetch showcase repositories for the portfolio UI.")
    limit: int = Field(default=DEFAULT_LIMIT, ge=1, le=50)
    use_agent: bool = Field(
        default=True,
        description="Use the LLM agent. Set false to call the GitHub tool directly (useful for tests).",
    )


class ShowcaseRepo(BaseModel):
    title: str
    description: Optional[str] = None
    readme: str
    url: HttpUrl
    owner: str
    stars: int = 0
    topics: List[str] = Field(default_factory=list)


class AgentResponse(BaseModel):
    repos: List[ShowcaseRepo]
    source: str
    raw_output: Optional[Any] = None


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
                repos.append(
                    ShowcaseRepo(
                        title=item.get("full_name", name),
                        description=item.get("description"),
                        readme=readme_text,
                        url=item.get("html_url"),
                        owner=owner,
                        stars=item.get("stargazers_count", 0),
                        topics=item.get("topics", []),
                    ).model_dump()
                )
        return repos
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(
            f"GitHub API error {exc.response.status_code}: {exc.response.text}"
        ) from exc
    except httpx.RequestError as exc:
        raise RuntimeError(f"GitHub request failed: {exc}") from exc


@tool
def fetch_showcase_repos(limit: int = DEFAULT_LIMIT) -> List[Dict[str, Any]]:
    """Fetch showcase repos with title, description, and README content."""
    return _fetch_showcase_repos(limit)


def build_agent_graph():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required to run the agent.")

    tools = [fetch_showcase_repos]
    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0,
    )
    system_prompt = (
        "You are a portfolio data agent. Always call the fetch_showcase_repos tool. "
        "Return only JSON in the format {'repos': [...]} where each repo includes "
        "title, description, readme, url, owner, stars, and topics."
    )
    return create_react_agent(
        llm,
        tools,
        messages_modifier=system_prompt,
    )


agent_graph = None


def get_agent_graph():
    global agent_graph
    if agent_graph is None:
        agent_graph = build_agent_graph()
    return agent_graph


def extract_repos_from_messages(messages: List[Any]) -> List[Dict[str, Any]]:
    # Prefer tool message output (raw tool result).
    for message in reversed(messages):
        if isinstance(message, ToolMessage):
            content = message.content
            if isinstance(content, list):
                return content
            if isinstance(content, str):
                try:
                    parsed = json.loads(content)
                    if isinstance(parsed, dict) and "repos" in parsed:
                        return parsed["repos"]
                    if isinstance(parsed, list):
                        return parsed
                except json.JSONDecodeError:
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
            agent_input = (
                f"{request.query} Limit results to {request.limit} and respond with "
                "JSON in the shape {'repos': [...]}, where each repo has title, "
                "description, readme, url, owner, stars, and topics."
            )
            result = await agent.ainvoke({"messages": [HumanMessage(content=agent_input)]})
            messages = result.get("messages", [])
            repos = extract_repos_from_messages(messages)
            if not repos:
                raise HTTPException(status_code=502, detail="Agent did not return repository data.")
            final_text = ""
            if messages:
                final_text = getattr(messages[-1], "content", "")
            return AgentResponse(
                repos=[ShowcaseRepo(**repo) for repo in repos],
                source="agent",
                raw_output=final_text,
            )
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    try:
        repos = _fetch_showcase_repos(request.limit)
        return AgentResponse(
            repos=[ShowcaseRepo(**repo) for repo in repos],
            source="tool",
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=str(exc)) from exc


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
