"""Research Agent - Standalone script for LangGraph deployment.

This module creates a deep research agent with custom tools and prompts
for conducting web research with strategic thinking and context management.
"""

from datetime import datetime

from langchain.chat_models import init_chat_model
from langchain_google_genai import ChatGoogleGenerativeAI
from deepagents import create_deep_agent

from prompts.prompt import (
    SALES_AGENT_INSTRUCTIONS,
    IINSTRUCTIONS,
    PRICING_AGENT_INSTRUCTIONS,
    TECHNICAL_AGENT_INSTRUCTIONS
)
from tools.tool import get_all_products, tavily_search, think_tool, get_pending_rfps, get_price, docling_convert

# Limits
max_concurrent_research_units = 3
max_researcher_iterations = 3

# Get current date
current_date = datetime.now().strftime("%Y-%m-%d")

# Combine orchestrator instructions (RESEARCHER_INSTRUCTIONS only for sub-agents)
# INSTRUCTIONS = (
#     RESEARCH_WORKFLOW_INSTRUCTIONS
#     + "\n\n"
#     + "=" * 80
#     + "\n\n"
#     + SUBAGENT_DELEGATION_INSTRUCTIONS.format(
#         max_concurrent_research_units=max_concurrent_research_units,
#         max_researcher_iterations=max_researcher_iterations,
#     )
# )

# Create research sub-agent
sales_subagent = {
    "name": "sales-agent",
    "description": "Delegate research to the sub-agent researcher. Only give this researcher one topic at a time.",
    "system_prompt": SALES_AGENT_INSTRUCTIONS.format(date=current_date),
    "tools": [get_pending_rfps, docling_convert],
}
technical_subagent = {
    "name": "technical-agent",
    "description": "Handles technical analysis of RFPs",
    "system_prompt":TECHNICAL_AGENT_INSTRUCTIONS,
    "tools": [get_all_products]
}
pricing_subagent = {
    "name": "pricing-agent",
    "description": "This is the pricing agent",
    "system_prompt": PRICING_AGENT_INSTRUCTIONS,
    "tools": [get_price, get_all_products]
}
# Model Gemini 3 
# model = ChatGoogleGenerativeAI(model="gemini-3-pro-preview", temperature=0.0)

# Model Claude 4.5

from langchain_openai import ChatOpenAI
from os import getenv
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(
    api_key=getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    model="mistralai/devstral-2512:free"
)
# model = init_chat_model(model="anthropic:claude-sonnet-4-5-20250929", temperature=0.0)

# Create the agent
agent = create_deep_agent(
    model=model,
    tools=[get_pending_rfps, docling_convert, get_all_products, get_price, tavily_search, think_tool],
    system_prompt=IINSTRUCTIONS,
    subagents=[sales_subagent, technical_subagent, pricing_subagent],
)