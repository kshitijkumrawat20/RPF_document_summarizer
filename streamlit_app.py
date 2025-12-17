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

st.set_page_config(page_title="RFP Analysis Agent", page_icon="🔍", layout="wide")

st.title("🔍 RFP Analysis Agent")
st.markdown("Welcome! I'm an RFP analysis agent that can analyze RFPs, match technical specifications, and provide pricing estimates.")

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
last_delegated_agent_history = None
for idx, message in enumerate(st.session_state.messages):
    if isinstance(message, dict):
        role = message.get("role")
        content = message.get("content")
        with st.chat_message(role):
            st.markdown(content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            # Track task delegations
            if hasattr(message, 'tool_calls') and message.tool_calls:
                for tool_call in message.tool_calls:
                    # Handle both dict and object access patterns
                    tool_name = tool_call.get('name') if isinstance(tool_call, dict) else getattr(tool_call, 'name', None)
                    
                    if tool_name == 'task':
                        # Extract args - try both dict and object attribute access
                        if isinstance(tool_call, dict):
                            args = tool_call.get('args', {})
                        else:
                            args = getattr(tool_call, 'args', {})
                        
                        # Get agent name from args
                        if isinstance(args, dict):
                            agent_name = args.get('agent', 'unknown')
                        else:
                            agent_name = getattr(args, 'agent', 'unknown')
                        
                        last_delegated_agent_history = agent_name
                        
                        if st.session_state.get('show_tool_calls', False):
                            agent_display = {
                                'sales-agent': '🔍 Sales Agent', 
                                'technical-agent': '⚙️ Technical Agent', 
                                'pricing-agent': '💰 Pricing Agent'
                            }.get(agent_name, agent_name)
                            st.info(f"🎯 Delegating to **{agent_display}**")
                    elif st.session_state.get('show_tool_calls', False):
                        st.info(f"🔧 Called tool: **{tool_name}**")
            # Show AI message content
            if message.content:
                st.markdown(message.content)
    elif isinstance(message, ToolMessage):
        # Only show tool messages if enabled
        if st.session_state.get('show_tool_calls', False):
            with st.chat_message("assistant"):
                if message.name == 'task':
                    # Show agent name instead of generic "Subagent Response"
                    agent_display_names = {
                        'sales-agent': '🔍 Sales Agent',
                        'technical-agent': '⚙️ Technical Agent',
                        'pricing-agent': '💰 Pricing Agent',
                    }
                    display_name = agent_display_names.get(last_delegated_agent_history, '🤖 Subagent')
                    st.success(f"**{display_name} Response:**\n\n{message.content[:500]}...")
                else:
                    st.info(f"🛠️ Tool **{message.name}**:\n\n{message.content[:300]}...")

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
        
        # Use the custom callback handler
        st_callback = get_streamlit_cb(thinking_container)
        
        try:
            config = {
                "configurable": {"thread_id": "research_thread_122"},
                "callbacks": [st_callback]
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
                task_delegations = []
                
                for msg in new_messages:
                    if isinstance(msg, ToolMessage):
                        tool_messages.append(msg)
                        # Check if this is a task delegation result (subagent response)
                        if msg.name == 'task':
                            task_delegations.append(msg)
                    elif isinstance(msg, AIMessage):
                        # Don't skip messages - we want to see all agent thinking
                        ai_responses.append(msg)
                    
                    # Add to session state for persistence
                    st.session_state.messages.append(msg)
                
                # Clear the status
                status_placeholder.empty()
                
                # Display the agent workflow - always show if there are messages
                with thinking_container.expander("💭 Agent Workflow & Communication", expanded=True):
                    step_num = 1
                    current_agent = "Main Agent"
                    last_delegated_agent = None
                    agent_display_names = {
                        'sales-agent': '🔍 Sales Agent',
                        'technical-agent': '⚙️ Technical Agent',
                        'pricing-agent': '💰 Pricing Agent',
                    }
                    
                    # File system tools to hide from UI
                    hidden_tools = {'read_file', 'write_file', 'list_files', 'delete_file', 'create_directory'}
                    
                    for msg in new_messages:
                        if isinstance(msg, AIMessage):
                            # Show main agent reasoning/thinking
                            if msg.content:
                                st.markdown(f"### 🤖 Step {step_num}: {current_agent} Thinking")
                                st.markdown(msg.content)
                                step_num += 1
                            
                            # Check if this is task delegation
                            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                                for tool_call in msg.tool_calls:
                                    # Handle both dict and object access patterns
                                    tool_name = tool_call.get('name') if isinstance(tool_call, dict) else getattr(tool_call, 'name', None)
                                    
                                    # Skip file system tools
                                    if tool_name in hidden_tools:
                                        continue
                                    
                                    if tool_name == 'task':
                                        # Main agent delegating to subagent
                                        # Extract args - try both dict and object attribute access
                                        if isinstance(tool_call, dict):
                                            args = tool_call.get('args', {})
                                        else:
                                            args = getattr(tool_call, 'args', {})
                                        
                                        # Get agent name and task from args
                                        if isinstance(args, dict):
                                            agent_name = args.get('agent', 'unknown')
                                            task_content = args.get('task', 'No task description')
                                        else:
                                            agent_name = getattr(args, 'agent', 'unknown')
                                            task_content = getattr(args, 'task', 'No task description')
                                        
                                        display_name = agent_display_names.get(agent_name, f'🤖 {agent_name}')
                                        last_delegated_agent = agent_name
                                        
                                        st.markdown(f"### 🎯 Step {step_num}: Main Agent → {display_name}")
                                        st.info(f"**Task Delegation:**\n\n{task_content}")
                                        step_num += 1
                                        # Update current agent
                                        current_agent = display_name
                                    else:
                                        # Other tool calls by current agent (non-file system)
                                        # Extract tool args - try both dict and object attribute access
                                        if isinstance(tool_call, dict):
                                            tool_args = tool_call.get('args', {})
                                        else:
                                            tool_args = getattr(tool_call, 'args', {})
                                        
                                        st.markdown(f"### 🔧 Step {step_num}: {current_agent} calls `{tool_name}`")
                                        if show_tool_calls and tool_args:
                                            with st.expander(f"View tool arguments", expanded=False):
                                                st.json(tool_args if isinstance(tool_args, dict) else str(tool_args))
                                        step_num += 1
                        
                        elif isinstance(msg, ToolMessage):
                            # Skip file system tool results
                            if msg.name in hidden_tools:
                                continue
                                
                            # Subagent response or tool result
                            if msg.name == 'task':
                                # This is a response from the delegated subagent
                                display_name = agent_display_names.get(last_delegated_agent, '🤖 Subagent')
                                
                                st.markdown(f"### ✅ Step {step_num}: {display_name} → Main Agent")
                                # Truncate subagent response to 1000 chars
                                content = msg.content[:1000] + ("..." if len(msg.content) > 1000 else "")
                                st.success(content)
                                step_num += 1
                                # Reset to main agent after subagent completes
                                current_agent = "Main Agent"
                                last_delegated_agent = None
                            else:
                                # Regular tool result (non-file system)
                                st.markdown(f"### 🛠️ Step {step_num}: {current_agent} - Tool `{msg.name}` Result")
                                if len(msg.content) < 500:
                                    st.code(msg.content, language="text")
                                else:
                                    with st.expander(f"View full {msg.name} output ({len(msg.content)} chars)", expanded=False):
                                        st.code(msg.content[:1000] + ("..." if len(msg.content) > 1000 else ""), language="text")
                                step_num += 1
                        
                        st.divider()
                
                # Always display the final response prominently
                if ai_responses:
                    # Get the last AI message with content
                    final_response = None
                    for msg in reversed(ai_responses):
                        if msg.content:
                            final_response = msg
                            break
                    
                    if final_response and final_response.content:
                        result_container.success("✅ Final Analysis Complete")
                        result_container.markdown("---")
                        result_container.markdown(final_response.content)
                    elif ai_responses:
                        # If no message with content, show that the workflow completed
                        result_container.info("✅ Workflow completed. Check the Agent Workflow section above for details.")
                
                # Show summary of tool calls if enabled
                if show_tool_calls and tool_messages:
                    with st.expander(f"🛠️ All Tool Calls ({len(tool_messages)} total)", expanded=False):
                        for idx, tool_msg in enumerate(tool_messages, 1):
                            st.markdown(f"**{idx}. {tool_msg.name}**")
                            st.code(tool_msg.content[:300] + ("..." if len(tool_msg.content) > 300 else ""), language="text")
                            st.divider()
            
        except Exception as e:
            import traceback
            st.error(f"An error occurred: {str(e)}")
            st.error("Full traceback:")
            st.code(traceback.format_exc())
