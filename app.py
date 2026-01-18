"""FastAPI app exposing a LangGraph-powered agent with a GitHub showcase tool."""

import json
import logging
import os
from typing import Annotated, Any, Dict, List, Optional, TypedDict

logging.basicConfig(level=logging.DEBUG)

import httpx
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field, HttpUrl

load_dotenv()

GITHUB_API_URL = "https://api.github.com"
DEFAULT_LIMIT = 20
GITHUB_USERNAME = os.getenv("GITHUB_USERNAME")
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a portfolio data agent. If the user greets or chats, respond conversationally without tools. "
    "Only call tools for explicit requests about repositories, projects, showcase items, or GitHub data. "
    "If the request is unclear, ask a short clarification question before using tools. "
    "Use the available tools to gather data when needed. "
    "Choose the most appropriate tool based on its description. "
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


def wants_repo_data(text: str) -> bool:
    if not text:
        return False
    lowered = text.lower()
    return any(keyword in lowered for keyword in REPO_QUERY_KEYWORDS)


class AgentState(TypedDict):
    messages: Annotated[List[Any], add_messages]
    repos: Optional[List[Dict[str, Any]]]


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
    """Fetch showcase repos with title, description, and README content.
    
    Returns complete repo data ready for direct consumption.
    """
    repos = _fetch_showcase_repos(limit)
    # Convert any HttpUrl objects to strings for JSON serialization
    serializable_repos = []
    for repo in repos:
        serializable_repo = repo.copy()
        if "url" in serializable_repo and serializable_repo["url"] is not None:
            serializable_repo["url"] = str(serializable_repo["url"])
        serializable_repos.append(serializable_repo)
    return serializable_repos

# Registry of tools that return complete data and don't need LLM post-processing
DIRECT_RETURN_TOOLS = {"fetch_showcase_repos"}


def build_agent_graph():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required to run the agent.")

    tools = [fetch_showcase_repos]
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
        model = llm_with_tools if wants_repo_data(last_user_text) else llm
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
    return graph.compile()


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
                    f"If you call fetch_showcase_repos, use limit {request.limit}."
                )
            logger.info("Agent invocation", extra={"limit": request.limit, "query": request.query})
            result = await agent.ainvoke({"messages": [HumanMessage(content=agent_input)]})
            logger.debug(f"Agent result: {result}")
            messages = result.get("messages", [])
            logger.debug(f"Messages: {messages}")
            tool_called = any(isinstance(message, ToolMessage) for message in messages)
            repos: List[Dict[str, Any]] = []
            if tool_called:
                repos = extract_repos_from_messages(messages)
                logger.debug(f"Extracted repos: {repos}")
                if not repos:
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
        repos = _fetch_showcase_repos(request.limit)
        return AgentResponse(
            repos=[ShowcaseRepo(**repo) for repo in repos],
            source="tool",
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=str(exc)) from exc


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
