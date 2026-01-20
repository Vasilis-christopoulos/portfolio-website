"""Test script to verify list-only fast path behavior."""
import asyncio
import logging

from langchain_core.messages import HumanMessage
from app import build_agent_graph

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    """Test the agent with list-only fast path behavior."""
    agent = build_agent_graph()
    
    query = "Fetch 2 showcase repositories"
    logger.info(f"Query: {query}")
    
    # Track the execution flow
    logger.info("=" * 80)
    logger.info("EXECUTION FLOW:")
    logger.info("=" * 80)
    
    result = await agent.ainvoke(
        {"messages": [HumanMessage(content=query)]},
        config={"configurable": {"thread_id": "test-direct-return"}},
    )
    
    messages = result.get("messages", [])
    logger.info(f"\nTotal messages in flow: {len(messages)}")
    logger.info(f"Plan: {result.get('plan')}")
    
    for i, msg in enumerate(messages):
        msg_type = type(msg).__name__
        logger.info(f"\n{i+1}. {msg_type}")
        
        if msg_type == "AIMessage":
            content_preview = str(msg.content)[:100] if msg.content else ""
            logger.info(f"   → Response: {content_preview}")

    repos = result.get("repos") or []
    logger.info(f"\n{'=' * 80}")
    logger.info(f"RESULT: Found {len(repos)} repositories")
    logger.info(f"{'=' * 80}")
    
    # Show expected vs actual flow
    logger.info("\n" + "=" * 80)
    logger.info("EXPECTED FLOW (list-only fast path):")
    logger.info("=" * 80)
    logger.info("START → planner (LLM) → retrieve → END")
    
    logger.info("\n" + "=" * 80)
    if len(messages) == 1:
        logger.info("✅ SUCCESS: No answer LLM call (list-only fast path)")
    elif len(messages) == 2:
        logger.info("⚠️  INFO: Answer LLM ran (non-list behavior)")
    else:
        logger.info(f"⚠️  UNEXPECTED: {len(messages)} messages")


if __name__ == "__main__":
    asyncio.run(main())
