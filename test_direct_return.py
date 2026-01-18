"""Test script to verify direct_return tool optimization."""
import asyncio
import logging

from langchain_core.messages import HumanMessage
from app import build_agent_graph, extract_repos_from_messages

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    """Test the agent with direct_return optimization."""
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
    
    for i, msg in enumerate(messages):
        msg_type = type(msg).__name__
        logger.info(f"\n{i+1}. {msg_type}")
        
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            logger.info(f"   → Calls tool: {msg.tool_calls[0]['name']}")
        
        if msg_type == "ToolMessage":
            content_preview = str(msg.content)[:100] if msg.content else ""
            logger.info(f"   → Returns: {content_preview}...")
        
        if msg_type == "AIMessage" and not msg.tool_calls:
            content_preview = str(msg.content)[:100] if msg.content else ""
            logger.info(f"   → Response: {content_preview}")
    
    repos = extract_repos_from_messages(messages)
    logger.info(f"\n{'=' * 80}")
    logger.info(f"RESULT: Found {len(repos)} repositories")
    logger.info(f"{'=' * 80}")
    
    # Show expected vs actual flow
    logger.info("\n" + "=" * 80)
    logger.info("EXPECTED FLOW (with direct_return):")
    logger.info("=" * 80)
    logger.info("START → agent (LLM) → tools → END")
    logger.info("                ↓")
    logger.info("         Decides to call tool")
    logger.info("                           ↓")
    logger.info("                    Fetches data")
    logger.info("                                    ↓")
    logger.info("                              Returns data (NO 2nd LLM call)")
    
    logger.info("\n" + "=" * 80)
    if len(messages) == 3:
        logger.info("✅ SUCCESS: Only 1 LLM call (optimized flow)")
    elif len(messages) == 4:
        logger.info("❌ WARNING: 2 LLM calls (old flow - agent processed tool result)")
    else:
        logger.info(f"⚠️  UNEXPECTED: {len(messages)} messages")


if __name__ == "__main__":
    asyncio.run(main())
