import streamlit as st
from agent import agent  # Import deep agent instead of app
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_core.callbacks.base import BaseCallbackHandler
import inspect
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
from streamlit.delta_generator import DeltaGenerator
from typing import Callable, TypeVar

# --- Fix for StreamlitCallbackHandler with LangGraph ---
def get_streamlit_cb(parent_container: DeltaGenerator) -> BaseCallbackHandler:
    """
    Creates a Streamlit callback handler that integrates fully with any LangChain ChatLLM integration,
    updating the provided Streamlit container with outputs such as tokens, model responses,
    and intermediate steps. This function ensures that all callback methods run within
    the Streamlit execution context, fixing the NoSessionContext() error commonly encountered
    in Streamlit callbacks.
    """
    fn_return_type = TypeVar('fn_return_type')

    def add_streamlit_context(fn: Callable[..., fn_return_type]) -> Callable[..., fn_return_type]:
        ctx = get_script_run_ctx()

        def wrapper(*args, **kwargs) -> fn_return_type:
            add_script_run_ctx(ctx=ctx)
            return fn(*args, **kwargs)

        return wrapper

    st_cb = StreamlitCallbackHandler(parent_container)

    for method_name, method_func in inspect.getmembers(st_cb, predicate=inspect.ismethod):
        if method_name.startswith('on_'):
            setattr(st_cb, method_name, add_streamlit_context(method_func))

    return st_cb
# -------------------------------------------------------

st.set_page_config(page_title="Deep Research Agent", page_icon="🔍", layout="wide")

st.title("🔍 Deep Research Agent")
st.markdown("Welcome! I'm a deep research agent that can analyze RFPs, match technical specifications, and provide pricing estimates.")

# Sidebar with agent info
with st.sidebar:
    st.header("🤖 Agent Capabilities")
    st.markdown("""
    **Main Agent:**
    - Coordinates sub-agents
    - Manages workflow
    - Synthesizes results
    
    **Sub-Agents:**
    - 🔍 **Sales Agent**: Scans and analyzes RFPs
    - ⚙️ **Technical Agent**: Matches product specs
    - 💰 **Pricing Agent**: Calculates pricing
    """)
    
    st.divider()
    
    # Show agent thinking toggle
    show_thinking = st.checkbox("Show Agent Reasoning", value=True)
    show_tool_calls = st.checkbox("Show Tool Calls", value=False)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "show_thinking" not in st.session_state:
    st.session_state.show_thinking = True
if "show_tool_calls" not in st.session_state:
    st.session_state.show_tool_calls = False

# Update session state from sidebar
st.session_state.show_thinking = show_thinking
st.session_state.show_tool_calls = show_tool_calls

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    if isinstance(message, dict):
        role = message.get("role")
        content = message.get("content")
        with st.chat_message(role):
            st.markdown(content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        # Skip tool call messages if show_tool_calls is False
        if hasattr(message, 'tool_calls') and message.tool_calls and not st.session_state.get('show_tool_calls', False):
            continue
        with st.chat_message("assistant"):
            st.markdown(message.content)
    # Skip ToolMessages unless show_tool_calls is True
    elif isinstance(message, ToolMessage):
        if st.session_state.get('show_tool_calls', False):
            with st.chat_message("assistant"):
                st.info(f"🛠️ Tool: {message.name}\n\n{message.content[:500]}...")  # Truncate long outputs

# Accept user input
if prompt := st.chat_input("What would you like to do?"):
    # Add user message to chat history
    st.session_state.messages.append(HumanMessage(content=prompt))
    
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # Create containers for different types of output
        status_placeholder = st.empty()
        thinking_container = st.container()
        result_container = st.container()
        
        try:
            config = {
                "configurable": {"thread_id": "research_thread_1"},
            }
            
            print(f"🚀 Starting deep agent with config: {config['configurable']}")
            
            # Show progress status
            status_placeholder.info("🚀 Starting agent workflow...")
            
            # Invoke the deep agent
            response = agent.invoke(
                {"messages": [HumanMessage(content=prompt)]},
                config=config
            )
            
            status_placeholder.success("✅ Agent workflow completed!")
            print(f"✅ Deep agent completed. Response keys: {response.keys() if response else 'None'}")
            
            # Process and display messages intelligently
            if response and "messages" in response:
                start_idx = len(st.session_state.messages)
                new_messages = response["messages"][start_idx:]
                
                # Separate different types of messages
                ai_responses = []
                tool_messages = []
                
                for msg in new_messages:
                    if isinstance(msg, ToolMessage):
                        tool_messages.append(msg)
                    elif isinstance(msg, AIMessage):
                        # Skip messages that are just tool calls without content
                        if hasattr(msg, 'tool_calls') and msg.tool_calls and not msg.content:
                            continue
                        ai_responses.append(msg)
                    
                    # Add to session state for persistence
                    st.session_state.messages.append(msg)
                
                # Display agent reasoning if enabled
                if show_thinking and len(ai_responses) > 1:
                    with thinking_container.expander("💭 Agent Reasoning Process", expanded=False):
                        for idx, msg in enumerate(ai_responses[:-1], 1):
                            st.markdown(f"**Step {idx}:**")
                            # Check if this looks like subagent output
                            if any(keyword in msg.content.lower() for keyword in ['sales-agent', 'technical-agent', 'pricing-agent', 'sub-agent', 'subagent']):
                                st.info(msg.content)
                            else:
                                st.markdown(msg.content)
                            st.divider()
                
                # Always display the final response
                if ai_responses:
                    final_response = ai_responses[-1]
                    
                    # Format the final response nicely
                    content = final_response.content
                    
                    # Check if response contains tables (markdown tables)
                    if '|' in content and '---' in content:
                        result_container.markdown(content)
                    else:
                        result_container.success("✅ Analysis Complete")
                        result_container.markdown(content)
                
                # Show tool calls if enabled
                if show_tool_calls and tool_messages:
                    with st.expander("🛠️ Tool Calls", expanded=False):
                        for tool_msg in tool_messages:
                            st.code(f"Tool: {tool_msg.name}\nResult: {tool_msg.content[:300]}...", language="text")
            
        except Exception as e:
            import traceback
            st.error(f"An error occurred: {str(e)}")
            st.error("Full traceback:")
            st.code(traceback.format_exc())
