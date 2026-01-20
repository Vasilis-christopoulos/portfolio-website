"""Debug script to test the agent locally."""
import asyncio
import json
import logging
import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from app import build_agent_graph, extract_repo_ids_from_messages, extract_repos_from_messages, load_repos_by_ids

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

load_dotenv()


async def main():
    """Run the agent and print detailed output."""
    logger.info("Building agent graph...")
    agent = build_agent_graph()
    
    query = "Fetch showcase repositories for the portfolio UI. Limit results to 5."
    logger.info(f"Running agent with query: {query}")
    
    result = await agent.ainvoke(
        {"messages": [HumanMessage(content=query)]},
        config={"configurable": {"thread_id": "debug"}},
    )
    
    print("\n" + "="*80)
    print("AGENT RESULT:")
    print("="*80)
    print(f"Keys in result: {result.keys()}")
    print(f"Repos in state: {result.get('repos')}")
    
    print("\n" + "="*80)
    print("MESSAGES:")
    print("="*80)
    messages = result.get("messages", [])
    for i, msg in enumerate(messages):
        print(f"\nMessage {i}: {type(msg).__name__}")
        content = getattr(msg, 'content', None)
        print(f"Content type: {type(content)}")
        if isinstance(content, str):
            print(f"Content (first 200 chars): {content[:200]}")
        else:
            print(f"Content: {content}")
        print(f"Tool calls: {getattr(msg, 'tool_calls', None)}")
        if hasattr(msg, 'tool_call_id'):
            print(f"Tool call ID: {msg.tool_call_id}")
    
    print("\n" + "="*80)
    print("EXTRACTED REPOS:")
    print("="*80)
    repo_ids = extract_repo_ids_from_messages(messages)
    repos = load_repos_by_ids(repo_ids) if repo_ids else extract_repos_from_messages(messages)
    print(json.dumps(repos, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())
