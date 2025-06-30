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
        padding-bottom: 120px;
    }
    .agent-dropdown-item {
        padding: 12px 16px;
        cursor: pointer;
        border-bottom: 1px solid #F3F4F6;
        display: flex;
        align-items: center;
        transition: background-color 0.2s ease;
    }
    .agent-dropdown-item:hover {
        background-color: #F8FAFC;
    }
    .agent-dropdown-item:last-child {
        border-bottom: none;
    }
    .agent-info {
        display: flex;
        align-items: center;
        font-weight: 600;
        color: #374151;
        margin-bottom: 8px;
        width: 100%;
        justify-content: space-between;
    }
    .agent-questions-count {
        font-size: 0.8rem;
        color: #6B7280;
        font-weight: normal;
    }
    .agent-questions {
        margin-left: 38px;
        margin-top: 8px;
    }
    .question-item {
        padding: 8px 12px;
        margin-bottom: 4px;
        background-color: #F9FAFB;
        border-radius: 8px;
        cursor: pointer;
        font-size: 0.9rem;
        color: #4B5563;
        transition: all 0.2s ease;
        border-left: 3px solid transparent;
        font-weight: normal;
    }
    .question-item:hover {
        background-color: #E0F2FE;
        border-left-color: #0EA5E9;
        transform: translateX(3px);
    }
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
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
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
if 'selected_question' not in st.session_state:
    st.session_state.selected_question = ""
if 'last_selected_agent' not in st.session_state:
    st.session_state.last_selected_agent = "Select an Agent"
if 'last_chosen_question' not in st.session_state:
    st.session_state.last_chosen_question = "Select a Questions"
if 'input_key' not in st.session_state:
    st.session_state.input_key = 0
if 'current_active_agent' not in st.session_state:
    st.session_state.current_active_agent = None
if 'agent_workflow_state' not in st.session_state:
    st.session_state.agent_workflow_state = {}

# Define agents with their questions
AGENTS_QUESTIONS = {
    "Insights Agent": [
        "What are the quarterly trends in total orders shipped from 2023 to 2025?",
        "Can you identify the top 3 customers by total pallets shipped?",
        "How does the average distance per shipment vary by product type?"
    ],
    "Dynamic Cost Optimizer": [
        "How can we optimize daily delivery schedules in January 2025 based on demand?",
        "Can we keep cost per order below £100 in 2025 through dynamic scheduling?",
        "How can we dynamically adjust delivery schedules for TESCO in February 2025 to minimize costs?"
    ],
    "Static Cost Optimizer": [
        "What savings result from maintaining 85% pallet utilization per truck?",
        "What cost savings can be achieved by consolidating shipments to postcode EN every 5 days in 2024?"
    ],
    "Pallet Utilization Optimization Agent": [
        "How many pallets were shipped in total for the product type \"AMBCONTROL\"?",
        "Provide total cost per pallet shipped in 2024?",
        "What is the average cost per pallet shipped 2024?"
    ]
}

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
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.last_input = ""
    st.session_state.code_snippets = []
    st.session_state.code_panel_open = True
    st.session_state.selected_question = ""
    st.session_state.last_selected_agent = "Select an Agent"
    st.session_state.last_chosen_question = "Select a Questions"
    st.session_state.input_key += 1
    st.session_state.current_active_agent = None
    st.session_state.agent_workflow_state = {}

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
        if st.button("📝 Code Analysis" if not st.session_state.code_panel_open else "✕ Hide Code", 
                     key="toggle_code_panel",
                     help="Toggle Code Analysis Panel",
                     use_container_width=True):
            st.session_state.code_panel_open = not st.session_state.code_panel_open
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

def display_sidebar():
    """Display the sidebar with information and options."""
    with st.sidebar:
        st.markdown("## About")
        st.markdown("""
        This application provides insights and optimization recommendations for UK distribution operations.
        ### Features:
        - Cost optimization analysis
        - Shipment consolidation recommendations
        - Pallet utilization optimization
        - Data-driven insights
        """)
        st.markdown("### Our Multi-Agent AI System")
        agent_cols = st.columns([1, 3])
        with agent_cols[0]:
            st.markdown('<span class="agent-icon supervisor-icon">S</span>', unsafe_allow_html=True)
        with agent_cols[1]:
            st.markdown("**Supervisor**<br>Routes queries to specialized agents", unsafe_allow_html=True)
        agent_cols = st.columns([1, 3])
        with agent_cols[0]:
            st.markdown('<span class="agent-icon optimizer-icon">D</span>', unsafe_allow_html=True)
        with agent_cols[1]:
            st.markdown("**Dynamic Optimizer**<br>Optimizes shipping costs dynamically", unsafe_allow_html=True)
        agent_cols = st.columns([1, 3])
        with agent_cols[0]:
            st.markdown('<span class="agent-icon optimizer-icon">S</span>', unsafe_allow_html=True)
        with agent_cols[1]:
            st.markdown("**Static Optimizer**<br>Analyzes static cost savings", unsafe_allow_html=True)
        agent_cols = st.columns([1, 3])
        with agent_cols[0]:
            st.markdown('<span class="agent-icon insights-icon">I</span>', unsafe_allow_html=True)
        with agent_cols[1]:
            st.markdown("**Insights**<br>Provides data analysis and visualizations", unsafe_allow_html=True)
        agent_cols = st.columns([1, 3])
        with agent_cols[0]:
            st.markdown('<span class="agent-icon pallet-icon">P</span>', unsafe_allow_html=True)
        with agent_cols[1]:
            st.markdown("**Pallet Utilization**<br>Optimizes pallet configurations", unsafe_allow_html=True)
        st.markdown("---")
        if st.button("🗑️ Clear Conversation", use_container_width=True):
            clear_conversation()
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
            <p>Ask a question that requires data analysis or visualization to see the code here!</p>
        </div>
        ''', unsafe_allow_html=True)
        return
    for i, pair in enumerate(st.session_state.conversation_pairs):
        messages, charts, code_snippets = pair
        if code_snippets:
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
        else:
            st.markdown(f'''
            <div class="code-analysis-section">
                <div class="code-analysis-header">
                    <span class="question-number">{i+1}</span>
                    Question {i+1} - No Code Generated
                </div>
                <div class="no-code-message" style="padding: 20px; margin: 10px 0;">
                    <p style="margin: 0; font-size: 0.9rem;">This question didn't require code analysis</p>
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

def display_message(message: Dict[str, str], is_user: bool = False):
    """Display a message in the chat interface."""
    if is_user:
        st.markdown(f'<div class="user-message">You: {message}</div>', unsafe_allow_html=True)
    else:
        agent = message.get("agent", "System")
        text = message.get("text", "")
        icon_class = "supervisor-icon"
        if "dynamic" in agent.lower():
            icon_class = "optimizer-icon"
        elif "static" in agent.lower():
            icon_class = "optimizer-icon"
        elif "insight" in agent.lower():
            icon_class = "insights-icon"
        elif "pallet" in agent.lower():
            icon_class = "pallet-icon"
        agent_icon = f'<span class="agent-icon {icon_class}">{agent[0].upper()}</span>'
        st.markdown(f'<div class="agent-message">{agent_icon}<span class="agent-name">{agent}:</span> {text}</div>', 
                   unsafe_allow_html=True)

def display_chart(chart_data: Dict[str, str]):
    """Display a chart from base64 encoded image data."""
    try:
        content = chart_data.get("content", "")
        if content:
            image_bytes = base64.b64decode(content)
            image = Image.open(io.BytesIO(image_bytes))
            with st.container():
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                st.image(image, use_container_width=400)
                st.markdown('</div>', unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error displaying chart: {e}")

def process_query(question: str):
    """Process a user query and update the UI."""
    if not question.strip():
        return
    st.session_state.messages.append({"text": question, "is_user": True})
    st.session_state.is_processing = True
    st.session_state.processing_steps = []
    st.session_state.last_input = ""
    st.session_state.selected_question = ""
    st.session_state.input_key += 1
    st.session_state.current_active_agent = None
    st.session_state.agent_workflow_state = {}
    st.rerun()

def add_real_agent_step(agent_key: str, action: str, details: str = "", status: str = "active"):
    """Add a real agent processing step based on actual workflow."""
    if agent_key not in AGENT_MAPPING:
        return
    
    agent_info = AGENT_MAPPING[agent_key]
    timestamp = time.strftime("%H:%M:%S")
    
    status_class = f"status-{status}"
    status_text = status.upper()
    
    # Determine processing step class based on agent
    step_class = "processing-supervisor" if agent_key == "supervisor" else f"processing-{agent_key.split('_')[0]}"
    
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
        "agent": agent_key,
        "status": status
    })
    
    # Update current active agent
    if status == "active":
        st.session_state.current_active_agent = agent_key
    elif status == "complete" and st.session_state.current_active_agent == agent_key:
        st.session_state.current_active_agent = None

def display_processing_steps(container):
    """Display all processing steps with real agent workflow."""
    for step in st.session_state.processing_steps:
        step_type = step["type"]
        content = step["content"]
        timestamp = step.get("timestamp", time.strftime("%H:%M:%S"))
        
        # Add real-time animation for active agents
        animation_class = ""
        if step.get("status") == "active":
            animation_class = " real-time-agent"
        elif step.get("status") == "thinking":
            animation_class = " agent-thinking"
        elif step.get("status") == "responding":
            animation_class = " agent-responding"
            
        container.markdown(
            f'<div class="processing-step {step_type}{animation_class}">{content}<span class="step-timestamp">{timestamp}</span></div>', 
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
            <div class="progress-percentage">{progress}%</div>
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
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None

def delete_conversation_for_thread_and_user(thread_id, user_id):
    """Delete conversation documents from MongoDB."""
    MONGODB_URI = "mongodb+srv://akshatasb0108:0u6NVe68ucJr4fP8@clusterai.lnrva2k.mongodb.net/?retryWrites=true&w=majority&appName=clusterai"
    DB_NAME = "checkpointing_db"
    client = MongoClient(MONGODB_URI)
    db = client[DB_NAME]
    collection = db["checkpoints_aio"]
    query = {
        "thread_id": thread_id,
        "metadata.user_id": bytes(f'"{user_id}"', 'utf-8')
    }
    result = collection.delete_many(query)
    st.write(f"Deleted {result.deleted_count} documents.")

async def process_agent_query(question: str, thread_id: str, user_id: str, llm, shipment_df, rate_card, insights_df, sku_master):
    """Process query through multi-agent graph with real workflow tracking."""
    MONGODB_URI = "mongodb+srv://akshatasb0108:0u6NVe68ucJr4fP8@clusterai.lnrva2k.mongodb.net/?retryWrites=true&w=majority&appName=clusterai"
    DB_NAME = "checkpointing_db"
    
    async with AsyncMongoDBSaver.from_conn_string(MONGODB_URI, db_name=DB_NAME) as checkpointer:
        multi_agent_graph = await create_agent_graph(
            llm=llm,
            shipment_df=shipment_df,
            rate_card=rate_card,
            insights_df=insights_df,
            SKU_master=sku_master,
            checkpointer=checkpointer
        )
        
        config = {"configurable": {"thread_id": thread_id, "user_id": user_id}}
        state = {
            "messages": [HumanMessage(content=question)],
            "next": "supervisor",
            "visual_outputs": [],
            "current_agent": None,
            "metadata": {},
            "parameters": None
        }
        
        if question.lower() == "erase memory":
            delete_conversation_for_thread_and_user(thread_id, user_id)
            yield {
                "event": "agent_step",
                "agent": "supervisor",
                "action": "Memory Cleared",
                "details": f"Deleted thread_id: {thread_id} and user_id: {user_id}",
                "status": "complete"
            }
            yield {
                "event": "message",
                "agent": "System",
                "text": f"Deleted current `thread_id`: {thread_id} and `user_id`: {user_id} from database..!"
            }
            yield {
                "event": "final_response",
                "status": "success"
            }
            return
        
        # Track the workflow state
        current_step = 0
        total_steps = 0
        
        async for current_state in multi_agent_graph.astream(state, config):
            if isinstance(current_state, dict):
                # Extract the current node/agent from the state
                node_name = list(current_state.keys())[0] if current_state else "unknown"
                section = list(current_state.values())[0] if current_state else {}
                
                # Map node names to our agent keys
                agent_key = "supervisor"
                if "insights" in node_name.lower():
                    agent_key = "insights_agent"
                elif "dynamic" in node_name.lower():
                    agent_key = "dynamic_cost_optimizer"
                elif "static" in node_name.lower():
                    agent_key = "static_cost_optimizer"
                elif "pallet" in node_name.lower():
                    agent_key = "pallet_utilization_agent"
                
                # Update workflow state
                if 'next' in current_state:
                    state['next'] = current_state['next']
                if 'parameters' in current_state:
                    state['parameters'] = current_state['parameters']
                
                # Emit agent step event
                if section and 'messages' in section and section['messages']:
                    message = section['messages'][0]
                    content = message.content
                    name = message.name or "System"
                    
                    # Determine action based on content
                    action = "Processing Query"
                    if "analyzing" in content.lower():
                        action = "Analyzing Data"
                    elif "calculating" in content.lower():
                        action = "Performing Calculations"
                    elif "generating" in content.lower():
                        action = "Generating Response"
                    elif "optimizing" in content.lower():
                        action = "Optimizing Parameters"
                    
                    yield {
                        "event": "agent_step",
                        "agent": agent_key,
                        "action": action,
                        "details": f"Processing: {content[:100]}..." if len(content) > 100 else content,
                        "status": "active"
                    }
                    
                    # Emit message event
                    yield {
                        "event": "message",
                        "agent": name,
                        "text": content
                    }
                    
                    # Mark agent as complete
                    yield {
                        "event": "agent_step",
                        "agent": agent_key,
                        "action": "Task Completed",
                        "details": "Response generated successfully",
                        "status": "complete"
                    }
                
                # Handle visual outputs
                if state.get('visual_outputs'):
                    for path in state['visual_outputs']:
                        image_base64 = encode_image_to_base64(path)
                        if image_base64:
                            yield {
                                "event": "chart",
                                "content": image_base64
                            }
                            state['visual_outputs'].remove(path)
        
        yield {
            "event": "final_response",
            "status": "success"
        }

async def stream_agent_response(question, thread_id, user_id, llm, shipment_df, rate_card, insights_df, sku_master, process_container, progress_container, chat_container):
    """Helper function to handle async streaming of agent responses with real workflow tracking."""
    new_messages = []
    new_charts = []
    conversation_code_snippets = {}
    progress = 10
    
    try:
        async for event in process_agent_query(
            question=question,
            thread_id=thread_id,
            user_id=user_id,
            llm=llm,
            shipment_df=shipment_df,
            rate_card=rate_card,
            insights_df=insights_df,
            sku_master=sku_master
        ):
            if event["event"] == "agent_step":
                # Add real agent workflow step
                add_real_agent_step(
                    agent_key=event.get("agent", "supervisor"),
                    action=event.get("action", "Processing"),
                    details=event.get("details", ""),
                    status=event.get("status", "active")
                )
                
                # Update progress based on agent activity
                if event.get("status") == "active":
                    progress += 5
                elif event.get("status") == "complete":
                    progress += 10
                
                # Display updated steps
                display_processing_steps(process_container)
                display_progress_bar(progress_container, min(progress, 90), f"Agent: {AGENT_MAPPING.get(event.get('agent', 'supervisor'), {}).get('name', 'Unknown')} - {event.get('action', 'Processing')}")
                
            elif event["event"] == "message":
                message_text = event.get("text", "")
                clean_message, code_snippets = extract_code_from_message(message_text)
                conversation_code_snippets.update(code_snippets)
                
                message_obj = {
                    "text": clean_message,
                    "agent": event.get("agent", "System"),
                    "is_user": False
                }
                st.session_state.messages.append(message_obj)
                new_messages.append(message_obj)
                
                with chat_container:
                    display_message(message_obj)
                
                progress += 5
                display_progress_bar(progress_container, min(progress, 90), f"Response from {message_obj['agent']}")
                
            elif event["event"] == "chart":
                chart_data = event.get("content", "")
                new_charts.append({"content": chart_data})
                st.session_state.charts.append({"content": chart_data})
                
                with chat_container:
                    display_chart({"content": chart_data})
                
                progress += 5
                display_progress_bar(progress_container, min(progress, 90), "Rendering visualization...")
                
            elif event["event"] == "final_response":
                status_text = "Analysis Complete!" if event.get("status") == "success" else "Error occurred"
                state = "complete" if event.get("status") == "success" else "error"
                return new_messages, new_charts, conversation_code_snippets, status_text, state
                
    except Exception as e:
        error_message = f"Error processing query: {str(e)}"
        message_obj = {
            "text": error_message,
            "agent": "System",
            "is_user": False
        }
        st.session_state.messages.append(message_obj)
        new_messages.append(message_obj)
        
        with chat_container:
            display_message(message_obj)
            
        # Add error step
        add_real_agent_step("supervisor", "Error Occurred", str(e), "error")
        display_processing_steps(process_container)
        
        return new_messages, new_charts, conversation_code_snippets, "Error occurred", "error"

def stream_response():
    """Stream the response from the multi-agent system with real workflow tracking."""
    if st.session_state.is_processing:
        with st.status("Processing Query in Multi-Agent System...", expanded=True) as status:
            process_container = st.empty()
            progress_container = st.empty()
            chat_container = st.container()
            
            # Get the last user message
            last_user_message_obj = None
            for msg in reversed(st.session_state.messages):
                if msg.get("is_user", False):
                    last_user_message_obj = msg
                    break
            
            if last_user_message_obj:
                with chat_container:
                    st.markdown(f'<div class="conversation-pair">', unsafe_allow_html=True)
                    st.markdown(f'''
                    <div style="margin-bottom: 15px;">
                        <span class="question-number">{len(st.session_state.conversation_pairs) + 1}</span>
                        <span style="font-weight: 600; color: #374151;">Question {len(st.session_state.conversation_pairs) + 1}</span>
                    </div>
                    ''', unsafe_allow_html=True)
                    display_message(last_user_message_obj["text"], is_user=True)
            
            # Initialize system
            display_progress_bar(progress_container, 5, "Initializing multi-agent system...")
            add_real_agent_step("supervisor", "System Initialization", "Setting up multi-agent workflow", "active")
            display_processing_steps(process_container)
            
            # Get API key and initialize
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                st.error("OPENAI_API_KEY not found")
                status.update(label="Error occurred", state="error")
                st.session_state.is_processing = False
                st.rerun()
            
            try:
                llm = get_supervisor_llm(api_key)
            except ValueError as e:
                st.error(f"Invalid API Key: {e}")
                status.update(label="Error occurred", state="error")
                st.session_state.is_processing = False
                st.rerun()
            
            # Load data
            add_real_agent_step("supervisor", "Loading Data", "Accessing shipment and rate data", "active")
            display_processing_steps(process_container)
            display_progress_bar(progress_container, 10, "Loading data...")
            
            shipment_df, rate_card, insights_df, sku_master = load_data()
            if shipment_df is None:
                st.error("Failed to load data")
                add_real_agent_step("supervisor", "Data Loading Failed", "Could not access required data files", "error")
                display_processing_steps(process_container)
                status.update(label="Error occurred", state="error")
                st.session_state.is_processing = False
                st.rerun()
            
            add_real_agent_step("supervisor", "Data Loaded Successfully", "All required data files loaded", "complete")
            display_processing_steps(process_container)
            
            # Run the async streaming logic with real workflow tracking
            new_messages, new_charts, conversation_code_snippets, status_text, state = asyncio.run(
                stream_agent_response(
                    question=last_user_message_obj["text"],
                    thread_id=st.session_state.thread_id,
                    user_id=st.session_state.user_id,
                    llm=llm,
                    shipment_df=shipment_df,
                    rate_card=rate_card,
                    insights_df=insights_df,
                    sku_master=sku_master,
                    process_container=process_container,
                    progress_container=progress_container,
                    chat_container=chat_container
                )
            )
            
            # Store conversation pair
            if last_user_message_obj:
                st.session_state.conversation_pairs.append((
                    [last_user_message_obj] + new_messages,
                    new_charts,
                    conversation_code_snippets
                ))
            else:
                st.session_state.conversation_pairs.append((
                    new_messages,
                    new_charts,
                    conversation_code_snippets
                ))
            
            # Final completion step
            add_real_agent_step("supervisor", "Workflow Complete", "All agents have completed their tasks", "complete")
            display_processing_steps(process_container)
            display_progress_bar(progress_container, 100, "Response ready!")
            
            status.update(label=status_text, state=state)
        
        st.session_state.is_processing = False
        st.rerun()

def display_fixed_input_area():
    """Display the fixed input area at the bottom."""
    input_container = st.container()
    with input_container:
        col_agent, col_question = st.columns([1, 3])
        
        with col_agent:
            agent_options = ["Select an Agent"] + list(AGENTS_QUESTIONS.keys())
            agent_index = agent_options.index(st.session_state.last_selected_agent)
            selected_agent = st.selectbox(
                "Agents",
                agent_options,
                index=agent_index,
                key="agent_selector_input",
                label_visibility="collapsed"
            )
        
        if selected_agent != st.session_state.last_selected_agent:
            st.session_state.last_selected_agent = selected_agent
            st.session_state.last_chosen_question = "Select a Question"
            st.session_state.selected_question = ""
            st.rerun()
        
        with col_question:
            questions_list = ["Select a Question"]
            if selected_agent != "Select an Agent":
                questions_list.extend(AGENTS_QUESTIONS[selected_agent])
            
            question_index = 0
            if st.session_state.last_chosen_question in questions_list:
                question_index = questions_list.index(st.session_state.last_chosen_question)
            
            chosen_question = st.selectbox(
                "Predefined Questions",
                questions_list,
                index=question_index,
                key="question_selector_input",
                label_visibility="collapsed"
            )
        
        if chosen_question != st.session_state.last_chosen_question:
            st.session_state.last_chosen_question = chosen_question
            if chosen_question != "Select a Question":
                st.session_state.selected_question = chosen_question
            else:
                st.session_state.selected_question = ""
            st.rerun()
        
        with st.form(key="input_form", clear_on_submit=True):
            col_input, col_button = st.columns([10, 1])
            
            with col_input:
                user_input = st.text_input(
                    "Ask a question",
                    value=st.session_state.selected_question,
                    placeholder="Select a question above or type your own here...",
                    key=f"main_input_{st.session_state.input_key}",
                    label_visibility="collapsed"
                )
            
            with col_button:
                submit_clicked = st.form_submit_button("⮞", use_container_width=True)
            
            if submit_clicked and user_input:
                process_query(user_input)
                st.session_state.selected_question = ""
                st.session_state.last_chosen_question = "Select a Question"

def main():
    """Main application function."""
    display_header()
    display_sidebar()
    
    if st.session_state.code_panel_open:
        main_col, panel_col = st.columns([2, 1])
    else:
        main_col = st.container()
    
    with main_col:
        st.markdown('<div class="main-content-wrapper">', unsafe_allow_html=True)
        
        # Stream response with real workflow tracking
        stream_response()
        
        # Display conversation history
        chat_container = st.container()
        with chat_container:
            if st.session_state.conversation_pairs:
                for i, pair in enumerate(st.session_state.conversation_pairs):
                    messages, charts, code_snippets = pair
                    with st.container():
                        st.markdown(f'<div class="conversation-pair">', unsafe_allow_html=True)
                        st.markdown(f'''
                        <div style="margin-bottom: 15px;">
                            <span class="question-number">{i+1}</span>
                            <span style="font-weight: 600; color: #374151;">Question {i+1}</span>
                        </div>
                        ''', unsafe_allow_html=True)
                        
                        for msg in messages:
                            if msg.get("is_user", False):
                                display_message(msg["text"], is_user=True)
                            else:
                                display_message(msg)
                        
                        for chart in charts:
                            display_chart(chart)
                        
                        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    if st.session_state.code_panel_open:
        with panel_col:
            display_code_panel()
    
    display_fixed_input_area()

if __name__ == "__main__":
    main()




