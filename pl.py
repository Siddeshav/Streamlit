import streamlit as st
import pandas as pd
import base64
import uuid
import time
import io
import asyncio
from PIL import Image
import os
import re
from typing import List, Dict, Any, Optional
from pymongo import MongoClient
from langchain_core.messages import HumanMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.mongodb.aio import AsyncMongoDBSaver
from dotenv import load_dotenv, find_dotenv
from src.utils.openai_api import get_supervisor_llm
from src.orchestrater.MultiAgentGraph import create_agent_graph
from config.config import SHIPMENT_DF_PATH, RATECARD_PATH, INSIGHTS_DATA_PATH, SKU_MASTER_PATH

# Load environment variables
_ = load_dotenv(find_dotenv())

# Set page configuration
st.set_page_config(
    page_title="Logistics Multi Agents AI System",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS (unchanged from original)
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1E3A8A;
        margin-bottom: 0.5rem;
    }
    .agent-name {
        font-weight: 600;
        color: #1E3A8A;
        font-size: 1.1rem;
    }
    .agent-message {
        background-color: #F3F4F6;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
    }
    .user-message {
        background-color: #E0F2FE;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
        border-left: 5px solid #0EA5E9;
    }
    .chart-container {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border: 1px solid #E5E7EB;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .stButton>button {
        background-color: #1E3A8A;
        color: white;
        border-radius: 5px;
        padding: 1rem 1rem;
        font-weight: 600;
    }
    .stButton>button:hover {
        background-color: #1E40AF;
    }
    .stTextInput>div>div>input {
        border-radius: 25px !important;
        border: 2px solid #E5E7EB !important;
        padding: 12px 16px !important;
        font-size: 16px !important;
    }
    .stTextInput>div>div>input:focus {
        border-color: #1E3A8A !important;
        box-shadow: 0 0 0 3px rgba(30, 58, 138, 0.1) !important;
    }
    .loading-spinner {
        display: flex;
        justify-content: center;
        margin: 20px 0;
    }
    .processing-step {
        padding: 15px;
        margin-bottom: 10px;
        border-radius: 8px;
        animation: fadeIn 0.6s ease-in-out;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border: 1px solid rgba(0,0,0,0.05);
        transition: all 0.3s ease;
    }
    .processing-step:hover {
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transform: translateY(-2px);
    }
    .processing-agent {
        background-color: #EFF6FF;
        border-left: 4px solid #3B82F6;
    }
    .processing-action {
        background-color: #F0FDF4;
        border-left: 4px solid #10B981;
    }
    .processing-complete {
        background-color: #F0FDF4;
        border-left: 4px solid #10B981;
    }
    .processing-wait {
        background-color: #FFFBEB;
        border-left: 4px solid #F59E0B;
    }
    .processing-error {
        background-color: #FEF2F2;
        border-left: 4px solid #EF4444;
    }
    .processing-supervisor {
        background-color: #EFF6FF;
        border-left: 4px solid #3B82F6;
    }
    .processing-insights {
        background-color: #FEF3C7;
        border-left: 4px solid #F59E0B;
    }
    .processing-optimizer {
        background-color: #ECFDF5;
        border-left: 4px solid #10B981;
    }
    .processing-pallet {
        background-color: #F3E8FF;
        border-left: 4px solid #8B5CF6;
    }
    .code-block {
        background-color: #1E1E1E;
        color: #FFFFFF;
        padding: 15px;
        border-radius: 8px;
        font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
        white-space: pre-wrap;
        font-size: 0.85rem;
        margin: 10px 0;
        overflow-x: auto;
        border: 1px solid #333;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    }
    .code-header {
        background-color: #2D2D2D;
        color: #FFFFFF;
        padding: 8px 15px;
        border-radius: 8px 8px 0 0;
        font-weight: 600;
        font-size: 0.9rem;
        border-bottom: 1px solid #444;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .code-container {
        background-color: #F8F9FA;
        border-radius: 8px;
        margin-bottom: 20px;
        border: 1px solid #E5E7EB;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .code-icon {
        width: 16px;
        height: 16px;
        background-color: #10B981;
        border-radius: 50%;
        display: inline-block;
    }
    .right-panel {
        background-color: #FAFAFA;
        border-left: 1px solid #E5E7EB;
        padding: 20px;
        min-height: 100vh;
    }
    .right-panel-header {
        font-size: 1.2rem;
        font-weight: 600;
        color: #374151;
        margin-bottom: 20px;
        padding-bottom: 10px;
        border-bottom: 2px solid #E5E7EB;
        position: sticky;
        top: 0;
        background-color: #FAFAFA;
        z-index: 10;
    }
    .agent-icon {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 28px;
        height: 28px;
        line-height: 28px;
        text-align: center;
        border-radius: 50%;
        color: white;
        font-weight: bold;
        margin-right: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.15);
    }
    .supervisor-icon {
        background: linear-gradient(135deg, #3B82F6, #1E40AF);
    }
    .optimizer-icon {
        background: linear-gradient(135deg, #10B981, #047857);
    }
    .insights-icon {
        background: linear-gradient(135deg, #F59E0B, #B45309);
    }
    .pallet-icon {
        background: linear-gradient(135deg, #8B5CF6, #6D28D9);
    }
    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    @keyframes pulse {
        0% {
            transform: scale(1);
            opacity: 1;
        }
        50% {
            transform: scale(1.05);
            opacity: 0.8;
        }
        100% {
            transform: scale(1);
            opacity: 1;
        }
    }
    .typing-indicator {
        display: inline-block;
        margin-right: 10px;
    }
    .typing-indicator span {
        display: inline-block;
        width: 8px;
        height: 8px;
        background-color: #3B82F6;
        border-radius: 50%;
        margin-right: 3px;
        animation: bounce 1.2s infinite ease-in-out;
    }
    .typing-indicator span:nth-child(2) {
        animation-delay: 0.2s;
        background-color: #10B981;
    }
    .typing-indicator span:nth-child(3) {
        animation-delay: 0.4s;
        background-color: #F59E0B;
    }
    @keyframes bounce {
        0%, 80%, 100% { transform: scale(0); }
        40% { transform: scale(1); }
    }
    .progress-container {
        margin: 15px 0;
    }
    .progress-bar {
        height: 6px;
        background-color: #E5E7EB;
        border-radius: 3px;
        overflow: hidden;
    }
    .progress-bar-fill {
        height: 100%;
        background: linear-gradient(90deg, #3B82F6, #10B981);
        border-radius: 3px;
        transition: width 0.4s ease;
    }
    .progress-status {
        display: flex;
        justify-content: space-between;
        font-size: 0.8rem;
        color: #6B7280;
        margin-top: 5px;
    }
    .progress-percentage {
        font-weight: 600;
        color: #1E3A8A;
    }
    .agent-status-active {
        animation: pulse 1.5s infinite;
    }
    .agent-header {
        display: flex;
        align-items: center;
        margin-bottom: 8px;
    }
    .agent-title {
        font-weight: 600;
        color: #1F2937;
    }
    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-left: 5px;
    }
    .status-active {
        background-color: #10B981;
        box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.2);
        animation: pulse 1.5s infinite;
    }
    .status-waiting {
        background-color: #F59E0B;
    }
    .status-complete {
        background-color: #10B981;
    }
    .status-text {
        font-size: 0.75rem;
        color: #6B7280;
        margin-left: 5px;
    }
    .agent-detail {
        margin-top: 5px;
        margin-left: 38px;
        padding-top: 5px;
        color: #4B5563;
        font-size: 0.9rem;
    }
    .step-timestamp {
        color: #6B7280;
        font-size: 0.75rem;
        float: right;
        font-family: monospace;
    }
    .no-code-message {
        text-align: center;
        color: #6B7280;
        font-style: italic;
        padding: 40px 20px;
        background-color: #F9FAFB;
        border-radius: 8px;
        border: 2px dashed #D1D5DB;
        margin: 20px 0;
    }
    .conversation-pair {
        margin-bottom: 40px;
        padding-bottom: 20px;
    }
    .conversation-pair:not(:last-child) {
        border-bottom: 1px solid #E5E7EB;
    }
    .code-analysis-section {
        background-color: #FAFAFA;
        border-radius: 8px;
        padding: 15px;
        margin: 20px 0;
        border: 1px solid #E5E7EB;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .code-analysis-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #374151;
        margin-bottom: 15px;
        padding-bottom: 8px;
        border-bottom: 2px solid #E5E7EB;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .input-container {
        margin-bottom: 1rem;
    }
    .question-number {
        background-color: #1E3A8A;
        color: white;
        border-radius: 50%;
        width: 24px;
        height: 24px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 0.8rem;
        font-weight: bold;
        margin-right: 8px;
    }
    .code-toggle-container {
        position: fixed;
        top: 120px;
        right: 20px;
        z-index: 1000;
    }
    .code-toggle-button {
        background-color: #1E3A8A !important;
        color: white !important;
        border-radius: 8px !important;
        padding: 0.5rem 1rem !important;
        font-weight: 600 !important;
        border: none !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
        transition: all 0.3s ease !important;
        cursor: pointer;
    }
    .code-toggle-button:hover {
        background-color: #1E40AF !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.15) !important;
    }
    .header-container {
        position: relative;
        width: 100%;
        margin-bottom: 20px;
    }
    .toggle-button-right {
        position: absolute;
        top: 0;
        right: 0;
        z-index: 100;
    }
    .main-content-wrapper {
        padding-bottom: 120px; /* Increased padding to ensure input is not overlapped */
    }
    /* Removed unused CSS classes: .agent-dropdown-item, .agent-info, .agent-questions-count, .agent-questions, .question-item */
    .real-time-agent {
        background: linear-gradient(45deg, #EFF6FF, #DBEAFE);
        border-left: 4px solid #3B82F6;
        animation: pulse 2s infinite;
    }
    .agent-thinking {
        background: linear-gradient(45deg, #FFFBEB, #FEF3C7);
        border-left: 4px solid #F59E0B;
    }
    .agent-responding {
        background: linear-gradient(45deg, #F0FDF4, #DCFCE7);
        border-left: 4px solid #10B981;
    }
    /* Styles for the new sidebar */
    div[data-testid="stRadio"] > label > div[data-testid="stMarkdownContainer"] > p {
        font-size: 1.1rem !important;
        font-weight: 600 !important;
    }
    div[data-testid="stRadio"] > div[data-testid="stWidgetLabel"] { /* Hide the default st.radio label */
        display: none;
    }
    div[data-testid="stRadio"] > div {
        gap: 0.5rem !important; /* Adjust gap between radio items */
    }
    div[data-testid="stRadio"] label { /* Target individual radio items */
        padding: 10px 12px !important;
        border-radius: 8px !important;
        margin-bottom: 5px !important; /* Space between items */
        transition: background-color 0.3s ease, color 0.3s ease !important;
        display: block !important; /* Make label take full width */
        border: 1px solid transparent; /* For hover/selected effect */
        cursor: pointer;
    }
    /* Style for the selected radio button */
    div[data-testid="stRadio"] input[type="radio"]:checked + div {
        background-color: #1E3A8A !important; /* Primary color */
        color: white !important;
        border-radius: 8px !important;
    }
    div[data-testid="stRadio"] input[type="radio"]:checked + div p {
        color: white !important;
    }
    /* Hover effect for non-selected radio buttons */
    div[data-testid="stRadio"] label:hover {
        background-color: #E0F2FE !important; /* Light blue hover */
    }
    div[data-testid="stRadio"] label:hover p {
        color: #1E3A8A !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = "Assist" # Default to Assist tab
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'charts' not in st.session_state:
    st.session_state.charts = []
if 'conversation_pairs' not in st.session_state:
    st.session_state.conversation_pairs = []
if 'thread_id' not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if 'user_id' not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())
if 'is_processing' not in st.session_state:
    st.session_state.is_processing = False
if 'processing_steps' not in st.session_state:
    st.session_state.processing_steps = []
if 'last_input' not in st.session_state:
    st.session_state.last_input = ""
if 'code_snippets' not in st.session_state:
    st.session_state.code_snippets = []
if 'code_panel_open' not in st.session_state:
    st.session_state.code_panel_open = True
# 'selected_question' is removed as its UI element (predefined questions dropdown) is gone.
# 'last_selected_agent' and 'last_chosen_question' are removed as their UI elements are gone.
if 'input_key' not in st.session_state:
    st.session_state.input_key = 0
if 'current_active_agent' not in st.session_state:
    st.session_state.current_active_agent = None
if 'agent_workflow_state' not in st.session_state:
    st.session_state.agent_workflow_state = {}
if 'search_query' not in st.session_state: # For search conversation bar
    st.session_state.search_query = ""

# Agent mapping for real workflow tracking
AGENT_MAPPING = {
    "supervisor": {"name": "Supervisor", "icon": "S", "class": "supervisor-icon"},
    "insights_agent": {"name": "Insights Agent", "icon": "I", "class": "insights-icon"},
    "dynamic_cost_optimizer": {"name": "Dynamic Cost Optimizer", "icon": "D", "class": "optimizer-icon"},
    "static_cost_optimizer": {"name": "Static Cost Optimizer", "icon": "S", "class": "optimizer-icon"},
    "pallet_utilization_agent": {"name": "Pallet Utilization Agent", "icon": "P", "class": "pallet-icon"}
}

# Helper Functions
def clear_conversation():
    """Clear all conversation data and start fresh."""
    st.session_state.messages = []
    st.session_state.charts = []
    st.session_state.conversation_pairs = []
    st.session_state.processing_steps = []
    st.session_state.is_processing = False
    st.session_state.thread_id = str(uuid.uuid4()) # Reset thread_id for a new conversation context
    st.session_state.last_input = ""
    st.session_state.code_snippets = []
    # st.session_state.code_panel_open = True # Decided by user toggle, not reset
    st.session_state.selected_question = ""
    # st.session_state.last_selected_agent = "Select an Agent" # Keep for assist page context
    # st.session_state.last_chosen_question = "Select a Questions" # Keep for assist page context
    st.session_state.input_key += 1
    st.session_state.current_active_agent = None
    st.session_state.agent_workflow_state = {}
    st.session_state.search_query = ""
    # st.session_state.active_tab = "Assist" # Or your preferred default after clearing

def display_header():
    """Display the application header with logo, title, and toggle button."""
    st.markdown('<div class="header-container">', unsafe_allow_html=True)
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown("""
            <div class="main-header">🚚 Logistics Multi Agents AI System</div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="toggle-button-right">', unsafe_allow_html=True)
        if st.session_state.active_tab == "Assist": # Only show toggle on Assist page
            if st.button("📝 Code Analysis" if not st.session_state.code_panel_open else "✕ Hide Code",
                         key="toggle_code_panel",
                         help="Toggle Code Analysis Panel",
                         use_container_width=True):
                st.session_state.code_panel_open = not st.session_state.code_panel_open
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

def display_sidebar():
    """Display the sidebar with navigation and options."""
    with st.sidebar:
        # Custom CSS for st.radio can be injected here or globally if preferred
        # For simplicity, added some specific CSS for radio buttons in global CSS block

        # st.image("path/to/your/logo.png", width=100) # Optional: Add a logo
        st.markdown("<h1 style='text-align: center; margin-bottom: 1rem;'>Your App</h1>", unsafe_allow_html=True) # Placeholder for app name/logo

        tabs = ["Discover", "Assist"]
        # Get current index for radio button
        try:
            current_tab_index = tabs.index(st.session_state.active_tab)
        except ValueError:
            current_tab_index = 1 # Default to Assist if active_tab is somehow invalid

        selected_tab = st.radio(
            "MENU", # This label will be hidden by CSS if stWidgetLabel display:none is used
            tabs,
            key="sidebar_tabs",
            index=current_tab_index,
            # label_visibility="collapsed" # Use CSS to hide if more control needed
        )
        if selected_tab != st.session_state.active_tab:
            st.session_state.active_tab = selected_tab
            st.rerun()

        st.markdown("---")
        if st.button("🗑️ Clear Conversation", use_container_width=True, key="clear_conv_sidebar"):
            clear_conversation()
            st.session_state.active_tab = "Assist" # Go to Assist tab after clearing
            st.rerun()

def extract_code_from_message(message_content):
    """Extract code snippets from message content."""
    calc_pattern = r'\*\*Code used for calculation:\*\*\n\`\`\`python\n(.*?)\n\`\`\`'
    viz_pattern = r'\*\*Code used for visualization:\*\*\n\`\`\`python\n(.*?)\n\`\`\`'
    calc_code = re.search(calc_pattern, message_content, re.DOTALL)
    viz_code = re.search(viz_pattern, message_content, re.DOTALL)
    code_snippets = {}
    clean_message = message_content
    if calc_code:
        code_snippets['calculation'] = calc_code.group(1).strip()
        clean_message = re.sub(calc_pattern, '', clean_message, flags=re.DOTALL)
    if viz_code:
        code_snippets['visualization'] = viz_code.group(1).strip()
        clean_message = re.sub(viz_pattern, '', clean_message, flags=re.DOTALL)
    clean_message = re.sub(r'\n\s*\n\s*\n', '\n\n', clean_message).strip()
    return clean_message, code_snippets

def display_code_panel():
    """Display the code panel on the right side."""
    st.markdown('<div class="right-panel-header">📝 Code Analysis</div>', unsafe_allow_html=True)
    if not st.session_state.conversation_pairs:
        st.markdown('''
        <div class="no-code-message">
            <p>💻 No code snippets available yet</p>
            <p>Ask a question on the 'Assist' tab that requires data analysis or visualization to see the code here!</p>
        </div>
        ''', unsafe_allow_html=True)
        return

    # Filter conversation pairs based on search query if active
    # This is a basic client-side filter. For large histories, server-side/DB search is better.
    filtered_pairs = st.session_state.conversation_pairs
    if st.session_state.get('search_query', ''):
        query = st.session_state.search_query.lower()
        filtered_pairs = [
            pair for pair in st.session_state.conversation_pairs
            if any(query in (msg.get("text","").lower() if isinstance(msg, dict) else msg.lower()) for msg in pair[0])
        ]
        if not filtered_pairs:
            st.markdown('<p style="text-align:center; color: #6B7280;">No matching conversations found for your search.</p>', unsafe_allow_html=True)
            return # Added return here

    for i, pair_data in enumerate(filtered_pairs):
        # Ensure pair_data is correctly unpacked
        if len(pair_data) == 3:
            messages, charts, code_snippets = pair_data
        else:
            # Fallback or error handling if structure is unexpected
            # For example, if code_snippets might be missing from older data
            messages, charts = pair_data[:2]
            code_snippets = {} # Assume no code snippets if not present
            # st.warning(f"Conversation pair {i+1} has unexpected structure.") # Optional warning

        if code_snippets: # Check if code_snippets is not None and not empty
            st.markdown(f'''
            <div class="code-analysis-section">
                <div class="code-analysis-header">
                    <span class="question-number">{i+1}</span>
                    Code Analysis for Question {i+1}
                </div>
            </div>
            ''', unsafe_allow_html=True)
            if 'calculation' in code_snippets:
                st.markdown('''
                <div class="code-container">
                    <div class="code-header">
                        <span class="code-icon"></span>
                        Code Used for Calculation
                    </div>
                </div>
                ''', unsafe_allow_html=True)
                st.code(code_snippets['calculation'], language='python')
            if 'visualization' in code_snippets:
                st.markdown('''
                <div class="code-container">
                    <div class="code-header">
                        <span class="code-icon"></span>
                        Code Used for Visualization
                    </div>
                </div>
                ''', unsafe_allow_html=True)
                st.code(code_snippets['visualization'], language='python')
        elif st.session_state.active_tab == "Assist": # Only show "No Code Generated" if on Assist tab and relevant
            st.markdown(f'''
            <div class="code-analysis-section">
                <div class="code-analysis-header">
                    <span class="question-number">{i+1}</span>
                    Question {i+1} - No Code Generated
                </div>
                <div class="no-code-message" style="padding: 20px; margin: 10px 0;">
                    <p style="margin: 0; font-size: 0.9rem;">This question didn't require code analysis or no code was extracted.</p>
                </div>
            </div>
            ''', unsafe_allow_html=True)


def encode_image_to_base64(path):
    """Convert an image file to base64 string."""
    try:
        with open(path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")
    except Exception as e:
        st.error(f"Error encoding image: {e}")
        return None

def display_message(message_data: Any, is_user: bool = False):
    """Display a message in the chat interface. Handles both string and dict."""
    if is_user:
        # User message is expected to be a string (the question)
        st.markdown(f'<div class="user-message">You: {message_data}</div>', unsafe_allow_html=True)
    else:
        # Agent message is expected to be a dict
        agent = message_data.get("agent", "System")
        text = message_data.get("text", "")

        icon_class = AGENT_MAPPING.get(agent, {}).get("class", "supervisor-icon") # Default to supervisor
        # More specific mapping if agent name from message_data doesn't directly match AGENT_MAPPING keys
        if "dynamic" in agent.lower(): icon_class = "optimizer-icon"
        elif "static" in agent.lower(): icon_class = "optimizer-icon"
        elif "insight" in agent.lower(): icon_class = "insights-icon"
        elif "pallet" in agent.lower(): icon_class = "pallet-icon"
        elif "supervisor" in agent.lower(): icon_class = "supervisor-icon"

        agent_icon_char = AGENT_MAPPING.get(agent, {}).get("icon", agent[0].upper() if agent else "S")

        st.markdown(f'<div class="agent-message"><span class="agent-icon {icon_class}">{agent_icon_char}</span><span class="agent-name">{agent}:</span> {text}</div>',
                   unsafe_allow_html=True)


def display_chart(chart_data: Dict[str, str]):
    """Display a chart from base64 encoded image data."""
    try:
        content = chart_data.get("content", "")
        if content:
            image_bytes = base64.b64decode(content)
            image = Image.open(io.BytesIO(image_bytes))
            # Make charts appear in the middle by controlling width
            st.image(image, use_container_width=False, width=600) # Adjust width as needed
            # Removed the chart-container div to let Streamlit handle centering better with width
    except Exception as e:
        st.error(f"Error displaying chart: {e}")

def process_query(question: str):
    """Process a user query and update the UI."""
    if not question.strip():
        return
    # Append user message as a simple string for user, dict for agent
    st.session_state.messages.append({"text": question, "is_user": True, "agent": "User"}) # Standardize message format
    st.session_state.is_processing = True
    st.session_state.processing_steps = []
    st.session_state.last_input = "" # Clear last input after processing
    # st.session_state.selected_question = "" # Clear selected if used
    st.session_state.input_key += 1
    st.session_state.current_active_agent = None
    st.session_state.agent_workflow_state = {}
    st.rerun()

def add_real_agent_step(agent_key: str, action: str, details: str = "", status: str = "active"):
    """Add a real agent processing step based on actual workflow."""
    # Ensure agent_key is valid or map it
    mapped_agent_key = agent_key.lower()
    if mapped_agent_key not in AGENT_MAPPING:
        # Try to infer from common names if not a direct key match
        if "insights" in mapped_agent_key: mapped_agent_key = "insights_agent"
        elif "dynamic" in mapped_agent_key: mapped_agent_key = "dynamic_cost_optimizer"
        elif "static" in mapped_agent_key: mapped_agent_key = "static_cost_optimizer"
        elif "pallet" in mapped_agent_key: mapped_agent_key = "pallet_utilization_agent"
        elif "supervisor" in mapped_agent_key: mapped_agent_key = "supervisor"
        else: # Fallback if no match
            # st.warning(f"Unknown agent key for step: {agent_key}")
            # Default to supervisor or a generic display
            default_agent_info = {"name": agent_key, "icon": agent_key[0].upper() if agent_key else "?", "class": "supervisor-icon"}
            agent_info = default_agent_info
            step_class = "processing-agent" # Generic step class
            # return # Optionally skip adding step for unknown agents

    if mapped_agent_key in AGENT_MAPPING:
        agent_info = AGENT_MAPPING[mapped_agent_key]
        step_class = "processing-supervisor" if mapped_agent_key == "supervisor" else f"processing-{mapped_agent_key.split('_')[0]}"

    timestamp = time.strftime("%H:%M:%S")
    status_class = f"status-{status}"
    status_text = status.upper()

    step_content = f"""
    <div class="agent-header">
        <span class="agent-icon {agent_info['class']}">{agent_info['icon']}</span>
        <span class="agent-title">{agent_info['name']}</span>
        <span class="status-indicator {status_class}"></span>
        <span class="status-text">{status_text}</span>
    </div>
    <div class="agent-detail">
        <b>{action}</b>
        {f'<br>{details}' if details else ''}
    </div>
    """

    st.session_state.processing_steps.append({
        "type": step_class,
        "content": step_content,
        "timestamp": timestamp,
        "agent": mapped_agent_key, # Use the mapped key
        "status": status
    })

    if status == "active":
        st.session_state.current_active_agent = mapped_agent_key
    elif status == "complete" and st.session_state.current_active_agent == mapped_agent_key:
        st.session_state.current_active_agent = None

def display_processing_steps(container):
    """Display all processing steps with real agent workflow."""
    for step in st.session_state.processing_steps:
        step_type_class = step.get("type", "processing-agent") # Default class
        content = step["content"]
        timestamp = step.get("timestamp", time.strftime("%H:%M:%S"))

        animation_class = ""
        if step.get("status") == "active": animation_class = " real-time-agent"
        elif step.get("status") == "thinking": animation_class = " agent-thinking"
        elif step.get("status") == "responding": animation_class = " agent-responding"

        container.markdown(
            f'<div class="processing-step {step_type_class}{animation_class}">{content}<span class="step-timestamp">{timestamp}</span></div>',
            unsafe_allow_html=True
        )

def display_progress_bar(container, progress: float, status_text: str = ""):
    """Display a progress bar."""
    progress_html = f"""
    <div class="progress-container">
        <div class="progress-bar">
            <div class="progress-bar-fill" style="width: {progress}%;"></div>
        </div>
        <div class="progress-status">
            <div>{status_text}</div>
            <div class="progress-percentage">{int(progress)}%</div>
        </div>
    </div>
    """
    container.markdown(progress_html, unsafe_allow_html=True)

def load_data():
    """Load necessary data files."""
    try:
        shipment_df = pd.read_excel(SHIPMENT_DF_PATH, sheet_name="Sheet1")
        rate_card = pd.read_excel(RATECARD_PATH)
        insights_df = pd.read_csv(INSIGHTS_DATA_PATH)
        sku_master = pd.read_csv(SKU_MASTER_PATH)
        return shipment_df, rate_card, insights_df, sku_master
    except FileNotFoundError as e:
        st.error(f"Data file not found: {e}. Please check config.py and ensure data files are present.")
        return None, None, None, None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None

def delete_conversation_for_thread_and_user(thread_id, user_id):
    """Delete conversation documents from MongoDB."""
    # Ensure these are loaded from .env or config
    MONGODB_URI = os.getenv("MONGODB_URI", "mongodb+srv://akshatasb0108:0u6NVe68ucJr4fP8@clusterai.lnrva2k.mongodb.net/?retryWrites=true&w=majority&appName=clusterai")
    DB_NAME = os.getenv("MONGODB_DB_NAME", "checkpointing_db")
    if not MONGODB_URI:
        st.error("MongoDB URI not configured.")
        return
    try:
        client = MongoClient(MONGODB_URI)
        db = client[DB_NAME]
        collection = db["checkpoints_aio"] # Collection name from LangGraph
        query = {
            "thread_id": thread_id,
            # "metadata.user_id": user_id # Assuming user_id is stored directly in metadata
            # LangGraph stores user_id inside metadata like: {'configurable': {'user_id': 'my_user'}}
            # The exact path might depend on how user_id is structured in your LangGraph config
            # For AsyncMongoDBSaver, it's often part of the thread_id or config.
            # Let's assume thread_id is sufficient if user_id is embedded or not used for deletion this way.
        }
        # If user_id is indeed in metadata.user_id as bytes:
        # query["metadata.user_id"] = bytes(f'"{user_id}"', 'utf-8') # Original way

        # Simpler deletion by thread_id if that's how checkpointer is configured primarily
        result = collection.delete_many(query)
        st.toast(f"Deleted {result.deleted_count} conversation segments from database for current session.", icon="🗑️")
    except Exception as e:
        st.error(f"Error deleting conversation from database: {e}")


async def process_agent_query(question: str, thread_id: str, user_id: str, llm, shipment_df, rate_card, insights_df, sku_master):
    """Process query through multi-agent graph with real workflow tracking."""
    MONGODB_URI = os.getenv("MONGODB_URI", "mongodb+srv://akshatasb0108:0u6NVe68ucJr4fP8@clusterai.lnrva2k.mongodb.net/?retryWrites=true&w=majority&appName=clusterai")
    DB_NAME = os.getenv("MONGODB_DB_NAME", "checkpointing_db")

    if not MONGODB_URI:
        yield {"event": "error", "message": "MongoDB URI not configured."}
        return

    async with AsyncMongoDBSaver.from_conn_string(MONGODB_URI, db_name=DB_NAME) as checkpointer:
        try:
            multi_agent_graph = await create_agent_graph(
                llm=llm,
                shipment_df=shipment_df,
                rate_card=rate_card,
                insights_df=insights_df,
                SKU_master=sku_master,
                checkpointer=checkpointer
            )
        except Exception as e:
            yield {"event": "error", "message": f"Failed to create agent graph: {str(e)}"}
            return

        config = {"configurable": {"thread_id": thread_id, "user_id": user_id}}
        # Initial state for the graph
        initial_graph_state = {
            "messages": [HumanMessage(content=question)],
            # "next": "supervisor", # 'next' might be managed internally by the graph structure
            # "visual_outputs": [], # Ensure this is part of your graph's state schema if used
            # "current_agent": None,
            # "metadata": {},
            # "parameters": None
        }

        if question.lower() == "erase memory":
            # This deletion should ideally happen before graph interaction if it's a command
            # For now, keeping it as a special case handled by the agent query processor
            delete_conversation_for_thread_and_user(thread_id, user_id) # This is synchronous
            yield {
                "event": "agent_step", "agent": "supervisor", "action": "Memory Cleared",
                "details": f"Requested memory erasure for thread_id: {thread_id}", "status": "complete"
            }
            yield {"event": "message", "agent": "System", "text": f"Memory for session {thread_id} has been cleared."}
            yield {"event": "final_response", "status": "success"}
            return

        current_visual_outputs = [] # Temp store for visual outputs from state

        async for event_part in multi_agent_graph.astream_events(initial_graph_state, config, version="v1"):
            kind = event_part["event"]
            tags = event_part.get("tags", [])
            name = event_part.get("name", "") # Name of the node/agent

            if kind == "on_chat_model_stream":
                chunk = event_part["data"]["chunk"]
                if chunk.content:
                    # This gives raw token stream. We might want to aggregate for display.
                    # For now, let's assume higher-level events handle full messages.
                    pass
            elif kind == "on_tool_end": # Example: if tools generate charts
                if event_part["data"].get("outputs") and "visual_outputs" in event_part["data"]["outputs"]:
                    for path in event_part["data"]["outputs"]["visual_outputs"]:
                        image_base64 = encode_image_to_base64(path)
                        if image_base64:
                            yield {"event": "chart", "content": image_base64}
                            if os.path.exists(path): os.remove(path) # Clean up temp image file

            elif kind == "on_chain_end" or kind == "on_llm_end": # More relevant for agent steps
                # Determine agent key from name or tags
                agent_key_inferred = name.lower() # Default to node name
                if "supervisor" in agent_key_inferred: agent_key_inferred = "supervisor"
                elif "insights" in agent_key_inferred: agent_key_inferred = "insights_agent"
                elif "dynamic" in agent_key_inferred: agent_key_inferred = "dynamic_cost_optimizer"
                elif "static" in agent_key_inferred: agent_key_inferred = "static_cost_optimizer"
                elif "pallet" in agent_key_inferred: agent_key_inferred = "pallet_utilization_agent"
                else: # Try to get from tags if available
                    for tag in tags:
                        if tag.startswith("agent:"):
                            agent_key_inferred = tag.split(":")[1]
                            break

                outputs = event_part["data"].get("output", {})
                messages = []
                if isinstance(outputs, dict):
                    messages = outputs.get("messages", [])
                elif isinstance(outputs, list) and all(isinstance(m, HumanMessage) for m in outputs): # if output is list of messages
                    messages = outputs

                if messages:
                    last_message = messages[-1] # Process the last message from the node
                    content = last_message.content
                    agent_name_from_msg = getattr(last_message, 'name', agent_key_inferred) # Use name if available in message

                    yield {
                        "event": "agent_step", "agent": agent_key_inferred,
                        "action": f"Processing via {name}",
                        "details": f"Node '{name}' completed.", "status": "active" # Intermediate step
                    }
                    # Check for visual outputs directly in message if structured that way
                    # This part depends heavily on how your graph passes visual output paths
                    if hasattr(last_message, 'additional_kwargs') and last_message.additional_kwargs.get('visual_outputs'):
                        for path in last_message.additional_kwargs['visual_outputs']:
                            image_base64 = encode_image_to_base64(path)
                            if image_base64:
                                yield {"event": "chart", "content": image_base64}
                                if os.path.exists(path): os.remove(path) # Clean up

                    yield {"event": "message", "agent": agent_name_from_msg or agent_key_inferred, "text": content}
                    yield {
                        "event": "agent_step", "agent": agent_key_inferred,
                        "action": "Task Completed",
                        "details": f"Node '{name}' finished generating response.", "status": "complete"
                    }

        # Check for visual_outputs in the final state if your graph accumulates them
        final_state = await multi_agent_graph.ainvoke(initial_graph_state, config)
        if final_state and isinstance(final_state, dict):
            final_visual_outputs = final_state.get("visual_outputs", [])
            if isinstance(final_visual_outputs, list): # Ensure it's a list
                for path in final_visual_outputs:
                    if path not in current_visual_outputs: # Avoid duplicates if handled by stream
                        image_base64 = encode_image_to_base64(path)
                        if image_base64:
                            yield {"event": "chart", "content": image_base64}
                            if os.path.exists(path): os.remove(path)
            # Extract final message if not already streamed
            final_messages = final_state.get("messages", [])
            if final_messages:
                # This logic might be redundant if on_chain_end captures all messages.
                # Verify based on actual graph behavior.
                pass


        yield {"event": "final_response", "status": "success"}


async def stream_agent_response(question, thread_id, user_id, llm, shipment_df, rate_card, insights_df, sku_master, process_container, progress_container, chat_container):
    """Helper function to handle async streaming of agent responses with real workflow tracking."""
    new_messages_for_pair = [] # Only agent messages for the current pair
    new_charts_for_pair = []
    conversation_code_snippets = {}
    progress = 10 # Initial progress

    try:
        async for event in process_agent_query(
            question=question, thread_id=thread_id, user_id=user_id, llm=llm,
            shipment_df=shipment_df, rate_card=rate_card, insights_df=insights_df, sku_master=sku_master
        ):
            if event["event"] == "error":
                st.error(event["message"])
                # Add error step to UI
                add_real_agent_step("supervisor", "System Error", event["message"], "error")
                display_processing_steps(process_container) # Update UI
                return [], [], {}, f"Error: {event['message']}", "error"

            if event["event"] == "agent_step":
                add_real_agent_step(
                    agent_key=event.get("agent", "supervisor"),
                    action=event.get("action", "Processing"),
                    details=event.get("details", ""),
                    status=event.get("status", "active")
                )
                if event.get("status") == "active": progress = min(progress + 15, 85)
                elif event.get("status") == "complete": progress = min(progress + 10, 90)

                # This can cause a lot of rerenders. Consider batching updates or using st.empty for parts.
                # For now, direct update:
                with process_container: # Assuming process_container is an st.empty() or similar
                    display_processing_steps(st.session_state.processing_steps) # Display all steps so far
                with progress_container:
                     display_progress_bar(progress, f"Agent: {AGENT_MAPPING.get(event.get('agent','supervisor'),{}).get('name','System')} - {event.get('action','Working')}")


            elif event["event"] == "message":
                message_text = event.get("text", "")
                clean_message, code_snippets = extract_code_from_message(message_text)
                conversation_code_snippets.update(code_snippets) # Collect all code snippets

                message_obj = {"text": clean_message, "agent": event.get("agent", "System"), "is_user": False}
                st.session_state.messages.append(message_obj) # Append to global messages
                new_messages_for_pair.append(message_obj) # Append to current pair's messages

                with chat_container: # Assuming chat_container is main chat area
                    display_message(message_obj) # Display immediately

                progress = min(progress + 10, 90)
                with progress_container:
                    display_progress_bar(progress, f"Response from {message_obj['agent']}")

            elif event["event"] == "chart":
                chart_content = event.get("content", "")
                chart_obj = {"content": chart_content}
                st.session_state.charts.append(chart_obj) # Append to global charts
                new_charts_for_pair.append(chart_obj) # Append to current pair's charts

                with chat_container: # Display chart immediately
                    display_chart(chart_obj)

                progress = min(progress + 5, 90) # Smaller increment for charts
                with progress_container:
                    display_progress_bar(progress, "Rendering visualization...")

            elif event["event"] == "final_response":
                status_text = "Analysis Complete!" if event.get("status") == "success" else "Processing Error"
                final_state = "complete" if event.get("status") == "success" else "error"
                # Ensure supervisor marks completion
                add_real_agent_step("supervisor", "Workflow Complete", "All tasks processed.", final_state)
                with process_container:
                    display_processing_steps(st.session_state.processing_steps)
                with progress_container:
                    display_progress_bar(100, status_text)

                return new_messages_for_pair, new_charts_for_pair, conversation_code_snippets, status_text, final_state

    except Exception as e:
        st.error(f"Critical error during agent response streaming: {str(e)}")
        error_message_obj = {"text": f"System Error: {str(e)}", "agent": "System", "is_user": False}
        # Add to global and current pair messages
        st.session_state.messages.append(error_message_obj)
        new_messages_for_pair.append(error_message_obj)

        with chat_container: display_message(error_message_obj)
        add_real_agent_step("supervisor", "Critical Error", str(e), "error")
        with process_container: display_processing_steps(st.session_state.processing_steps)
        with progress_container: display_progress_bar(100, "Critical Error Occurred") # Show 100% on error to stop spinner
        return new_messages_for_pair, new_charts_for_pair, conversation_code_snippets, "Critical Error Occurred", "error"

    # Fallback if loop finishes without final_response (should not happen with robust agent_query)
    add_real_agent_step("supervisor", "Workflow Ended", "Processing finished.", "complete")
    with process_container: display_processing_steps(st.session_state.processing_steps)
    with progress_container: display_progress_bar(100, "Processing Ended")
    return new_messages_for_pair, new_charts_for_pair, conversation_code_snippets, "Processing Ended", "complete"


def stream_response():
    """Stream the response from the multi-agent system with real workflow tracking."""
    if st.session_state.is_processing:
        # Use st.expander for collapsable processing details
        with st.expander("🤖 Agent Activity Log...", expanded=True):
            process_container = st.container() # For individual steps
            progress_container = st.container() # For progress bar

        # Main chat area for messages and charts (outside the expander)
        chat_display_area = st.container()

        last_user_message_obj = None
        # Find the last actual user message to process
        for msg_info in reversed(st.session_state.messages):
            if msg_info.get("is_user"):
                last_user_message_obj = msg_info
                break

        if not last_user_message_obj:
            st.warning("No user query found to process.")
            st.session_state.is_processing = False
            return # Or st.rerun() if appropriate

        # Display user's question immediately in the chat area (if not already part of conversation_pairs)
        # This depends on whether process_query adds it to a displayable list or if stream_response should.
        # For now, assume conversation_pairs handles prior messages.

        # Initialize UI for processing
        with progress_container: display_progress_bar(5, "Initializing multi-agent system...")
        with process_container:
            st.session_state.processing_steps = [] # Clear steps for new query
            add_real_agent_step("supervisor", "System Initialization", "Setting up workflow", "active")
            display_processing_steps(st.session_state.processing_steps) # Show initial step

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            st.error("OPENAI_API_KEY not found in environment.")
            add_real_agent_step("supervisor", "Config Error", "OpenAI API Key missing.", "error")
            with process_container: display_processing_steps(st.session_state.processing_steps)
            st.session_state.is_processing = False
            return

        try:
            llm = get_supervisor_llm(api_key)
        except ValueError as e: # Assuming get_supervisor_llm might raise this for bad key
            st.error(f"Invalid API Key: {e}")
            add_real_agent_step("supervisor", "Config Error", f"Invalid API Key: {e}", "error")
            with process_container: display_processing_steps(st.session_state.processing_steps)
            st.session_state.is_processing = False
            return

        with process_container:
            add_real_agent_step("supervisor", "Loading Data", "Accessing datasets", "active")
            display_processing_steps(st.session_state.processing_steps)
        with progress_container: display_progress_bar(10, "Loading data...")

        shipment_df, rate_card, insights_df, sku_master = load_data()
        if shipment_df is None: # load_data now handles st.error
            add_real_agent_step("supervisor", "Data Loading Failed", "Required data files missing or corrupt.", "error")
            with process_container: display_processing_steps(st.session_state.processing_steps)
            st.session_state.is_processing = False
            return

        with process_container:
            add_real_agent_step("supervisor", "Data Loaded", "Datasets ready for analysis.", "complete")
            display_processing_steps(st.session_state.processing_steps)

        # Run async streaming
        new_agent_messages, new_charts, conversation_code_snippets, status_text, final_state = asyncio.run(
            stream_agent_response(
                question=last_user_message_obj["text"], # Pass only the text
                thread_id=st.session_state.thread_id,
                user_id=st.session_state.user_id,
                llm=llm,
                shipment_df=shipment_df, rate_card=rate_card, insights_df=insights_df, sku_master=sku_master,
                process_container=process_container, # For steps
                progress_container=progress_container, # For progress bar
                chat_container=chat_display_area # For messages and charts
            )
        )

        # After streaming, update conversation_pairs with the full exchange
        # The user message is already in st.session_state.messages
        # We need to form the pair: (list_of_all_messages_for_this_turn, list_of_charts_for_this_turn, code_snippets_for_this_turn)
        # The user message that triggered this is last_user_message_obj

        current_turn_messages = [last_user_message_obj] + new_agent_messages
        st.session_state.conversation_pairs.append((
            current_turn_messages,
            new_charts, # Charts from this turn
            conversation_code_snippets # Code from this turn
        ))

        st.session_state.is_processing = False
        # No st.rerun() here, let Streamlit flow naturally after async finishes.
        # The UI should have updated progressively. A final rerun might be needed if elements are not updating.
        # Forcing a rerun to ensure UI consistency after all async operations.
        st.rerun()


def display_fixed_input_area():
    """Display the fixed input area at the bottom for the Assist page."""
    # This input area will be styled with CSS to be fixed at the bottom right.
    # For simplicity in Streamlit, we'll place it at the bottom of the main flow.
    # True fixed positioning often requires more complex HTML/CSS injection.

    st.markdown("""
    <div class="fixed-input-container-wrapper">
        <div class="fixed-input-container">
    """, unsafe_allow_html=True)

    # The form helps manage input submission
    with st.form(key="logistic_chat_form", clear_on_submit=True):
        user_input = st.text_input(
            "Type your message to Logistic Chat...",
            key=f"logistic_chat_input_{st.session_state.input_key}", # Unique key for re-render
            label_visibility="collapsed",
            placeholder="Ask about logistics..."
        )
        # submit_button_col, _ = st.columns([1,3]) # To make button smaller
        # with submit_button_col:
        submit_clicked = st.form_submit_button("➤ Send", use_container_width=True)

        if submit_clicked and user_input.strip():
            process_query(user_input) # This will set is_processing and rerun
            # user_input is cleared due to clear_on_submit=True

    st.markdown("</div></div>", unsafe_allow_html=True)

    # Add CSS for the fixed input area
    st.markdown("""
    <style>
        .fixed-input-container-wrapper {
            position: fixed;
            bottom: 0;
            right: 0;
            padding: 15px 25px; /* Adjust padding as needed */
            background-color: #f0f2f6; /* A light background to distinguish */
            width: 350px; /* Adjust width of the input area */
            box-shadow: 0px -2px 10px rgba(0,0,0,0.1); /* Optional shadow */
            border-top-left-radius: 10px; /* Optional styling */
            z-index: 999;
        }
        .fixed-input-container .stTextInput input {
            border-radius: 15px !important; /* More rounded input field */
             border: 1px solid #ccc !important;
        }
        .fixed-input-container .stButton button {
            border-radius: 15px !important;
            background-color: #1E3A8A !important;
            color: white !important;
            width: 100% !important;
        }
    </style>
    """, unsafe_allow_html=True)

# --- Page Implementations ---
def display_discover_page():
    """Displays the Discover page content."""
    st.markdown("<h1 class='sub-header'>Discover Our Logistics AI System</h1>", unsafe_allow_html=True)

    st.markdown("""
    Welcome to the Logistics Multi-Agent AI System! This platform is designed to revolutionize your logistics operations
    by providing powerful insights, dynamic optimization, and intelligent assistance. Our suite of specialized AI agents
    works collaboratively to tackle complex logistical challenges, helping you make data-driven decisions, reduce costs,
    and improve efficiency.
    """)

    st.markdown("---")
    st.markdown("<h2 class='sub-header'>Meet Our Specialist Agents</h2>", unsafe_allow_html=True)

    agents_info = {
        "supervisor": {"name": "🎓 Supervisor Agent", "desc": "The central coordinator of the AI system. The Supervisor analyzes your queries and routes them to the most appropriate specialist agent or combination of agents to ensure you get the most accurate and relevant response. It manages the workflow and a orchestrates the collaboration between other agents."},
        "insights_agent": {"name": "📊 Insights Agent", "desc": "Your go-to agent for deep data analysis and trend identification. The Insights Agent processes historical and real-time data to uncover patterns, generate reports, and create visualizations. Ask it about shipment trends, customer behaviors, performance metrics, and more to gain a comprehensive understanding of your operations."},
        "dynamic_cost_optimizer": {"name": "⚙️ Dynamic Cost Optimizer", "desc": "This agent focuses on real-time cost optimization strategies. It can analyze current conditions, demand forecasts, and resource availability to suggest optimal routing, scheduling, and resource allocation to minimize operational costs dynamically. Ideal for questions about adjusting to changing scenarios."},
        "static_cost_optimizer": {"name": "💰 Static Cost Optimizer", "desc": "Analyzes cost-saving opportunities based on more stable parameters and strategic changes. Use this agent to explore potential savings from long-term strategies like shipment consolidation, network redesign, pallet utilization policies, or changes in operational rules."},
        "pallet_utilization_agent": {"name": "📦 Pallet Utilization Agent", "desc": "Specializes in optimizing how pallets are used within your shipments and warehouse. This agent can help analyze pallet fill rates, suggest better packing configurations, calculate costs per pallet, and identify inefficiencies in pallet handling and storage."}
    }

    for key, info in agents_info.items():
        icon_html = ""
        if key in AGENT_MAPPING: # Get icon from existing map
            agent_map_info = AGENT_MAPPING[key]
            icon_html = f'<span class="agent-icon {agent_map_info["class"]}" style="font-size: 1.5em; padding: 5px; margin-right:15px;">{agent_map_info["icon"]}</span>'

        st.markdown(f"""
        <div style="display: flex; align-items: center; margin-bottom: 15px;">
            {icon_html}
            <h3 style="margin-bottom:0;">{info['name']}</h3>
        </div>
        <p style="margin-left: 55px;">{info['desc']}</p>
        """, unsafe_allow_html=True)
        st.markdown("---")

    st.markdown("<h2 class='sub-header'>How to Use the 'Assist' Tab</h2>", unsafe_allow_html=True)
    st.markdown("""
    The **Assist** tab is your primary interface for interacting with our AI agents. Here’s how to get the best results:
    1.  **Ask Clear Questions:** Use the chat input at the bottom right (labeled "logistic chat") to ask specific questions about your logistics operations. The more detail you provide, the better the agents can assist you.
    2.  **Review Responses:** Agents will provide text-based answers. The Insights Agent may also generate charts and visualizations, which will appear directly in the chat flow.
    3.  **Code Analysis:** For queries that involve data manipulation or visualization, the "Code Analysis" panel (toggleable from the top right button on the Assist page) will show the Python code generated by the agents. This offers transparency and allows for verification.
    4.  **Search Conversations:** Use the "Search conversation" bar at the top of the Assist page to quickly find past topics or questions within your current session. (Note: Full historical search is a planned feature).
    5.  **Clear Conversation:** If you want to start fresh, use the "Clear Conversation" button in the sidebar. This will reset the chat history for the current session.

    Our AI system is continuously learning and improving. We value your feedback to make it even more powerful!
    """)


def display_assist_page():
    """Displays the Assist page (main chat interface)."""

    # Search bar (placeholder for now, or basic filter)
    search_query = st.text_input(
        "Search conversation...",
        key="search_conversation_input",
        value=st.session_state.search_query,
        placeholder="Filter current session chat history..."
    )
    if search_query != st.session_state.search_query:
        st.session_state.search_query = search_query
        st.rerun() # Rerun to apply filter in display_code_panel and potentially chat history

    st.markdown('<div class="main-content-wrapper">', unsafe_allow_html=True)

    # Stream response handles its own UI updates within containers passed to it.
    # It will create an expander for agent activity and a main area for chat.
    if st.session_state.is_processing:
        stream_response() # This function now manages its own UI containers for steps/progress

    # Display full conversation history (filtered if search_query is active)
    # This needs to be outside the is_processing block to show history when not processing.
    chat_history_container = st.container()
    with chat_history_container:
        if st.session_state.conversation_pairs:
            displayed_pairs = 0
            for i, pair_data in enumerate(reversed(st.session_state.conversation_pairs)): # Show newest first
                # Unpack pair_data carefully
                messages_in_pair, charts_in_pair, _ = pair_data # Code snippets not directly shown here

                # Apply search filter to messages in this pair
                show_pair = True
                if st.session_state.search_query:
                    query_lower = st.session_state.search_query.lower()
                    match_found = False
                    for msg_content in messages_in_pair:
                        # msg_content can be a string (user) or dict (agent)
                        text_to_search = ""
                        if isinstance(msg_content, dict):
                            text_to_search = msg_content.get("text", "").lower()
                        elif isinstance(msg_content, str):
                            text_to_search = msg_content.lower()

                        if query_lower in text_to_search:
                            match_found = True
                            break
                    if not match_found:
                        show_pair = False

                if show_pair:
                    displayed_pairs +=1
                    st.markdown(f'<div class="conversation-pair">', unsafe_allow_html=True)
                    # Display Question Number based on original index, not reversed
                    original_index = len(st.session_state.conversation_pairs) - 1 - i
                    st.markdown(f'''
                    <div style="margin-bottom: 15px; border-top: 1px solid #eee; padding-top:10px;">
                        <span class="question-number">{original_index + 1}</span>
                        <span style="font-weight: 600; color: #374151;">Interaction {original_index + 1}</span>
                    </div>
                    ''', unsafe_allow_html=True)

                    for msg_content in messages_in_pair:
                        is_user = isinstance(msg_content, dict) and msg_content.get("is_user", False)
                        display_message(msg_content.get("text") if is_user else msg_content, is_user=is_user)

                    for chart_obj in charts_in_pair:
                        display_chart(chart_obj)

                    st.markdown('</div>', unsafe_allow_html=True) # End conversation-pair

            if st.session_state.search_query and displayed_pairs == 0:
                st.markdown('<p style="text-align:center; color: #6B7280; margin-top: 20px;">No conversation turns match your search criteria.</p>', unsafe_allow_html=True)
        elif not st.session_state.is_processing : # Only show if not empty and not processing
             st.info("No conversation yet. Ask a question using the 'logistic chat' input below!")


    st.markdown('</div>', unsafe_allow_html=True) # End main-content-wrapper

    # The fixed input area for "logistic chat"
    display_fixed_input_area()


def main():
    """Main application function."""
    display_header() # Displays main app title and code analysis toggle
    display_sidebar() # Handles navigation between Discover/Assist

    # Determine main content area vs code panel
    if st.session_state.active_tab == "Assist" and st.session_state.code_panel_open:
        main_col, panel_col = st.columns([2, 1])
    else:
        main_col = st.container() # Use a single column if panel is closed or not on Assist
        panel_col = None # No panel column

    with main_col:
        if st.session_state.active_tab == "Discover":
            display_discover_page()
        elif st.session_state.active_tab == "Assist":
            display_assist_page()
            # Note: display_fixed_input_area is called within display_assist_page
            # to ensure it's only shown on the Assist page.

    if panel_col: # If panel_col was created (i.e., on Assist tab and panel is open)
        with panel_col:
            st.markdown('<div class="right-panel">', unsafe_allow_html=True)
            display_code_panel()
            st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
