from langchain_core.tools import tool
from pathlib import Path
import hashlib
from docling.document_converter import DocumentConverter


from langchain_openai import ChatOpenAI
from os import getenv
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(
    api_key=getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    model="nvidia/nemotron-3-nano-30b-a3b:free",
    streaming=True
)
# lets define the technical and pricing subgents tools 

## technical subagents tools
import pandas as pd

@tool
def get_all_products() -> str:
    """Lookup product specifications in the product catalog."""
    # For demonstration, returning a static response
    df = pd.read_csv("artifacts/Product_datasheet.csv")
    markdown = df.to_markdown()
    return markdown

@tool
def get_price() -> str:
    """Lookup product specifications in the product catalog."""
    # For demonstration, returning a static response
    df = pd.read_csv("artifacts/product_price.csv")
    markdown = df.to_markdown()
    return markdown


# # Example usage
# response = llm.invoke("What NFL team won the Super Bowl in the year Justin Bieber was born?")
# print(response.content)

@tool 
def get_pending_rfps() -> str:
    """Returns a list of RFPs due in next 3 months from the document repository."""
    # For demonstration, returning a static list

    pending_rfps = [{"source":"https://drive.google.com/uc?export=download&id=1pVqPmP_dQxXaOVrJA48Zzg8NP5oem-i_", "due_date":"20-12-2025", "status": "open"}]

    return str(pending_rfps)


@tool
def docling_convert(source: str) -> str:
    """Convert PDF/document to markdown and SAVE to filesystem.
    
    Args:
        source: URL or file path
        
    Returns: FILENAME where markdown is saved (use read_file to access content)
    """
    try:
        converter = DocumentConverter()
        result = converter.convert(source)
        markdown = result.document.export_to_markdown()
        print(markdown)
        ## save the markdown to file 
        return markdown
        # # Create unique filename in agent's workspace
        # filename = f"/research/doc_{Path(source).stem}.md"
        
        # # Agent will call write_file(filename, markdown) next
        # # Return ONLY filename to keep context tiny
        # return f"SAVED: {filename}\n(Use read_file('{filename}') to access full content)"
        
    except Exception as e:
        return f"ERROR: {str(e)}"
    # try:
    #     converter = DocumentConverter()
    #     result = converter.convert(source)
    #     markdown = result.document.export_to_markdown()
    #     doc_id = hashlib.md5(source.encode()).hexdigest()[:8]
    #     filename = f"/research/rfp_{doc_id}.md"
    #     return f"✅ SAVED: {filename}\n📏 {len(markdown)} chars\n➡️ read_file('{filename}')"
    # except Exception as e:
    #     return f"❌ ERROR: {str(e)}"


from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from datetime import datetime
from deepagents.middleware import FilesystemMiddleware

from deepagents import create_deep_agent

todays_date = datetime.now().date()

store = InMemoryStore()
checkpointer = MemorySaver()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage],add_messages]

backend_factory = lambda rt: CompositeBackend(
    default=StateBackend(rt),
    routes={"/memories/": StoreBackend(rt)}
)

# Use with thread_id for short-term memory persistence
def sales_agent_node(state: AgentState, config): 
    print("🔵 SALES AGENT NODE - Starting")
    print(f"   Config: {config.get('configurable', {})}")
    try:
        sales_agent = create_deep_agent(
            model = model,
            system_prompt="""You are a sales expert analyzing RFPs. 
            1. Scan the RFPs using `get_pending_rfps` tool
            2. Choose the most relevant RFP which is due in next 3 months, todays date is {todays_date}
            3. Convert the RFP document to markdown using `docling_convert` tool, it will return a whole markdown
            4. write the whole exact markdown in rfp_document.txt
            5. Use `write_todos` to plan analysis, dont make more than 3 steps. 
            6. Write analysis to /memories/sales_analysis.txt
            7. Give the response to main agent like : "sales analysis saved to /memories/sales_analysis.txt"
            """.format(todays_date=todays_date),
            checkpointer = checkpointer,
            store = store,
            tools = [get_pending_rfps, docling_convert],
            backend=backend_factory

        )
        print("🔵 SALES AGENT - Created successfully")
        
        response = sales_agent.invoke(
            {"messages": state['messages']},
            config
        )
        print(f"🔵 SALES AGENT NODE - Complete. Messages: {len(response['messages'])}")
        return {
            "messages": response["messages"]  # LangGraph merges with add_messages
        }
    except Exception as e:
        import traceback
        print(f"❌ SALES AGENT NODE - ERROR: {e}")
        print(traceback.format_exc())
        raise

## lets define the Technical and Pricing subagents 

technical_subagent = {
    "name": "technical-agent",
    "description": "Handles technical analysis of RFPs",
    "system_prompt":"""
You are a Technical Evaluation Agent for an OEM wire & cable manufacturer.

You will receive:
- A summary of an RFP
- The full RFP document (markdown) you can read at rfp_document.txt.
- Access to a repository of OEM product datasheets using tool 'get_all_products'.

Your tasks:
1. Extract and summarize all products listed in the Scope of Supply from the RFP.
2. For EACH product in scope of supply:
   a) Identify required technical specifications from the RFP.
   b) Search the OEM product datasheet repository.
   c) Recommend TOP 3 OEM product SKUs that best match the RFP specs.

3. For each recommended OEM SKU:
   - Calculate a "Spec Match %" where:
     - All RFP specs have equal weightage
     - Spec Match % = (Matched Specs / Total Required Specs) * 100

4. Create a COMPARISON TABLE for each RFP product:
   Columns must include:
   - RFP Spec Parameter
   - RFP Required Value
   - OEM Product 1 Spec Value
   - OEM Product 2 Spec Value
   - OEM Product 3 Spec Value

5. Based on Spec Match %, select ONE final OEM product SKU per RFP item.

6. Prepare a FINAL OUTPUT TABLE with:
   - RFP Product Name
   - Quantity
   - Selected OEM SKU
   - Spec Match %

7. Save the FINAL OUTPUT TABLE in /memories/technical_evaluation.md
   Send message to 
   - Main Agent - "technical evaluation saved to /memories/technical_evaluation.md"

Constraints:
- Do NOT consider pricing.
- Do NOT assume missing specs.
- Do NOT invent OEM products.
- If no product matches ≥70%, explicitly mention "Low Spec Match – Custom SKU Required".
""",
    "tools": [get_all_products]
}

## pricing subagent 
pricing_subagent = {
    "name": "pricing-agent",
    "description": "This is the pricing agent",
    "system_prompt": """
You are a Pricing Agent for a B2B OEM RFP response system.

You will receive:
1. A summary of testing & acceptance requirements from the Main Agent.
2. A final product recommendation table from the Technical Agent by reading the file /memories/technical_evaluation.md.

Your tasks:
1. For each recommended OEM product SKU:
   - Assign a UNIT MATERIAL PRICE using a dummy product pricing table.

2. For each testing & acceptance requirement:
   - Assign a TEST PRICE using a dummy services pricing table.

3. For EACH product in scope of supply:
   - Calculate:
     a) Total material cost
     b) Total testing & services cost
     c) Combined total cost

4. Prepare a CONSOLIDATED PRICING TABLE with columns:
   - RFP Product
   - OEM SKU
   - Quantity
   - Unit Price
   - Material Cost
   - Testing Cost
   - Total Cost

5. Send ONLY the consolidated pricing table to the Main Agent and also create the file saving the whole analysis /memories/pricing_analysis.md .

Constraints:
- Pricing must be deterministic and table-driven.
- Do NOT modify product selections.
- Do NOT perform technical validation.
- Assume dummy prices are accurate for estimation purposes.
"""
,
    "tools": [get_price]
}

## lets create the main agent

def main_agent_node(state: AgentState, config): 
    print("🟢 MAIN AGENT NODE - Starting")
    print(f"   Config: {config.get('configurable', {})}")
    try:
        main_agent = create_deep_agent(
            model = model,
            system_prompt="""
You are the Main Orchestrator Agent for a B2B RFP response system for an OEM
wire & cable manufacturer.

Your responsibilities:
1. Receive the selected RFP summary and RFP markdown document from the Sales Agent.
2. Prepare TWO contextual summaries:
   a) Technical Summary → for Technical Agent
   b) Pricing & Testing Summary → for Pricing Agent

3. Send the Technical Summary + RFP document to the Technical Agent.
4. Send the Pricing & Testing Summary to the Pricing Agent.

5. Receive:
   - Product recommendation table from Technical Agent
   - Consolidated pricing table from Pricing Agent

6. Consolidate the final RFP response which MUST include:
   - Final selected OEM product SKUs
   - Spec match percentage justification
   - Material prices
   - Testing & acceptance test costs
   - Total price per item in scope of supply

7. Ensure outputs are structured using:
   - Bullet summaries
   - Clean tables (no long paragraphs)

8. Start the conversation and explicitly end the conversation once the final RFP
   response is prepared.

Constraints:
- Do NOT invent products, specs, or prices.
- Use only data received from worker agents.
- Do not perform technical matching or pricing yourself.
""",
            checkpointer = checkpointer,
            store = store,
            subagents = [technical_subagent, pricing_subagent],
            backend=backend_factory
        )   
        print("🟢 MAIN AGENT - Created successfully")
        
        response = main_agent.invoke(
            {"messages": state['messages']},
            config
        )
        print(f"🟢 MAIN AGENT NODE - Complete. Messages: {len(response['messages'])}")
        return {
            "messages": response["messages"]  # LangGraph merges with add_messages
        }
    except Exception as e:
        import traceback
        print(f"❌ MAIN AGENT NODE - ERROR: {e}")
        print(traceback.format_exc())
        raise
from IPython.display import Image, display
graph = StateGraph(AgentState) 
graph.add_node("sales_agent_node", sales_agent_node)
graph.add_node("main_agent_node", main_agent_node)

graph.add_edge(START, "sales_agent_node")
graph.add_edge("sales_agent_node", "main_agent_node")
graph.add_edge("main_agent_node", END)
# app = graph.compile(checkpointer=checkpointer, store=store)

## display the graph 
# display(Image(app.get_graph().draw_mermaid_png(max_retries=5, retry_delay=2.0)))

# Add this to END of app.py (after graph compilation)
# ADD THIS at END of app.py (after graph.compile())
app = graph.compile(checkpointer=checkpointer, store=store)  # ✅ TOP LEVEL

if __name__ == "__main__":
    print("✅ RFP Graph ready!")