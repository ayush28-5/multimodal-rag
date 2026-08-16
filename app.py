import streamlit as st
import os
import sys
import shutil
import html as html_lib
import time
import requests
from urllib.parse import urlparse, unquote
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import (
    DATA_DIR, EXTRACTED_IMAGES_DIR, CHROMA_DB_DIR,
    LLM_MODEL, EMBEDDING_MODEL,
    ENABLE_STREAMING, ENABLE_PROFILING,
)
from src.embedder import repo
from src.rag import ask, ask_stream
from src.llm_provider import warmup_models
from src.conversation import (
    make_user_message,
    make_assistant_message,
    format_history_for_prompt,
    clear_conversation,
)
from src.query_rewriter import rewrite_question


SHOW_SOURCES_EXPANDER = False


st.set_page_config(
    page_title="Atlas — AI Document Assistant",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================
# SESSION STATE
# ============================================
for key, default in [
    ("messages", []),
    ("pending_files", []),
    ("ingest_status", None),
    ("last_upload_id", None),
    ("trigger_prompt", None),
    ("confirm_clear_kb", False),
    ("indexed_meta", {}),
    ("models_warmed", False),
    ("index_state", "ready"),
    ("last_doc_name", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


if not st.session_state.models_warmed:
    try:
        warmup_models()
        st.session_state.models_warmed = True
    except Exception as e:
        print(f"   [Warmup] Failed (will lazy-load): {e}")
        st.session_state.models_warmed = False


# ============================================
# CSS
# ============================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&display=swap');

* { font-family: 'Inter', -apple-system, sans-serif; }

:root {
    --bg: #FAFBFC;
    --bg-2: #F4F5F7;
    --surface: #FFFFFF;
    --surface-2: #F8F9FB;
    --surface-hover: #F0F2F5;
    --border: #E4E7EB;
    --border-soft: #EDEFF2;

    --text-1: #1A1D23;
    --text-2: #4A5159;
    --text-3: #7A8088;
    --text-4: #A8AEB5;

    --accent: #6366F1;
    --accent-soft: #4F46E5;
    --accent-pale: #818CF8;
    --accent-bg: rgba(99, 102, 241, 0.08);
    --accent-bg-2: rgba(99, 102, 241, 0.04);

    --user-bubble: #6366F1;
    --user-text: #FFFFFF;
    --ai-bubble: #FFFFFF;
    --ai-text: #1A1D23;

    --success: #16A34A;
    --success-bg: rgba(22, 163, 74, 0.08);
    --warning: #D97706;
    --warning-bg: rgba(217, 119, 6, 0.08);
    --danger: #DC2626;
    --danger-bg: rgba(220, 38, 38, 0.08);

    --shadow: 0 1px 3px rgba(15, 23, 42, 0.06);
    --shadow-lg: 0 8px 24px rgba(15, 23, 42, 0.08);
}

.stApp { background: var(--bg); }
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }

/* ============================================
   SIDEBAR LOCKED OPEN — collapse button hidden
   ============================================ */
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarHeader"] button,
button[kind="header"],
button[kind="headerNoPadding"],
[data-testid="baseButton-headerNoPadding"],
[data-testid="baseButton-header"] {
    display: none !important;
    visibility: hidden !important;
    opacity: 0 !important;
    pointer-events: none !important;
    width: 0 !important;
    height: 0 !important;
    padding: 0 !important;
    margin: 0 !important;
    border: none !important;
}

[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
    min-width: 270px !important;
    max-width: 270px !important;
    width: 270px !important;
    transform: translateX(0) !important;
    visibility: visible !important;
    margin-left: 0 !important;
    left: 0 !important;
    position: relative !important;
}

section[data-testid="stSidebar"] > div {
    width: 270px !important;
    transform: translateX(0) !important;
}

[data-testid="stSidebar"][aria-expanded="false"] {
    transform: translateX(0) !important;
    margin-left: 0 !important;
    visibility: visible !important;
}

/* ====== KILL TOP SPACE INSIDE SIDEBAR ====== */
[data-testid="stSidebar"] .block-container {
    padding: 0 14px 20px 14px !important;
    margin-top: 0 !important;
}

/* Remove the empty header strip Streamlit adds above the sidebar content */
[data-testid="stSidebarHeader"] {
    padding: 0 !important;
    margin: 0 !important;
    height: 0 !important;
    min-height: 0 !important;
    display: none !important;
}

[data-testid="stSidebarUserContent"] {
    padding-top: 0 !important;
    margin-top: 0 !important;
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 0 !important;
    margin-top: 0 !important;
}

/* First element flush to top */
[data-testid="stSidebar"] .block-container > div:first-child,
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] > div:first-child {
    margin-top: 0 !important;
    padding-top: 0 !important;
}

.brand {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 8px 2px 12px 2px;
    margin: 0 0 4px 0;
    border-bottom: 1px solid var(--border-soft);
}

.brand-mark {
    width: 32px;
    height: 32px;
    border-radius: 9px;
    background: var(--accent);
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-weight: 800;
    font-size: 0.85rem;
    flex-shrink: 0;
    box-shadow: 0 2px 8px rgba(99, 102, 241, 0.2);
}

.brand-info { flex: 1; min-width: 0; }

.brand-name {
    font-size: 0.88rem;
    font-weight: 700;
    color: var(--text-1);
    line-height: 1.15;
    letter-spacing: -0.01em;
}

.brand-sub {
    font-size: 0.62rem;
    color: var(--text-3);
    margin-top: 1px;
    font-weight: 500;
}

[data-testid="stSidebar"] .stButton button[kind="primary"] {
    background: var(--accent) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 7px 12px !important;
    font-weight: 600 !important;
    font-size: 0.78rem !important;
    text-transform: none !important;
    letter-spacing: 0 !important;
    width: 100% !important;
    box-shadow: 0 2px 6px rgba(99, 102, 241, 0.18) !important;
    transition: all 0.15s ease !important;
    text-align: left !important;
    min-height: 0 !important;
}

[data-testid="stSidebar"] .stButton button[kind="primary"]:hover {
    background: var(--accent-soft) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 3px 10px rgba(99, 102, 241, 0.25) !important;
}

[data-testid="stSidebar"] .stButton button[kind="secondary"] {
    background: transparent !important;
    color: var(--text-2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    padding: 6px 12px !important;
    font-weight: 500 !important;
    font-size: 0.76rem !important;
    text-transform: none !important;
    letter-spacing: 0 !important;
    width: 100% !important;
    box-shadow: none !important;
    text-align: left !important;
    min-height: 0 !important;
}

[data-testid="stSidebar"] .stButton button[kind="secondary"]:hover {
    background: var(--surface-hover) !important;
    color: var(--accent) !important;
    border-color: var(--accent) !important;
}

.sec-h {
    font-size: 0.6rem;
    font-weight: 700;
    color: var(--accent);
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin: 14px 2px 6px 2px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.sec-h .count {
    background: var(--accent-bg);
    color: var(--accent);
    font-size: 0.58rem;
    padding: 1px 6px;
    border-radius: 100px;
    font-weight: 700;
}

[data-testid="stFileUploader"] section {
    border: 1.5px dashed var(--border) !important;
    border-radius: 9px !important;
    background: var(--surface-2) !important;
    padding: 12px 8px !important;
    text-align: center;
    transition: all 0.15s ease;
}

[data-testid="stFileUploader"] section:hover {
    border-color: var(--accent) !important;
    background: var(--accent-bg-2) !important;
}

[data-testid="stFileUploader"] section > div {
    color: var(--text-2) !important;
    font-size: 0.72rem !important;
    font-weight: 500 !important;
}

[data-testid="stFileUploader"] section small {
    color: var(--text-3) !important;
    font-size: 0.62rem !important;
}

[data-testid="stFileUploader"] button { display: none !important; }

.stats-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin: 2px 2px 6px 2px;
}

.stats-row .label {
    font-size: 0.6rem;
    font-weight: 700;
    color: var(--accent);
    text-transform: uppercase;
    letter-spacing: 1.2px;
}

.stats-row .index-pill {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    font-size: 0.68rem;
    color: var(--text-1);
    font-weight: 600;
}

.stats-row .index-pill .dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    flex-shrink: 0;
}

.stats-row .index-pill .dot.ready {
    background: var(--success);
    box-shadow: 0 0 0 2px var(--success-bg);
}

.stats-row .index-pill .dot.processing {
    background: var(--warning);
    box-shadow: 0 0 0 2px var(--warning-bg);
    animation: pulseDot 1.5s ease-in-out infinite;
}

.stats-row .index-pill .dot.failed {
    background: var(--danger);
    box-shadow: 0 0 0 2px var(--danger-bg);
}

@keyframes pulseDot {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}

.stats-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 6px;
    margin: 4px 0 10px 0;
}

.stat-tile {
    background: var(--surface);
    border: 1px solid var(--border-soft);
    border-radius: 8px;
    padding: 8px 6px;
    text-align: center;
    transition: all 0.15s ease;
    position: relative;
    overflow: hidden;
}

.stat-tile::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 2px;
    background: var(--accent);
    opacity: 0.7;
}

.stat-tile:hover {
    border-color: var(--accent);
    transform: translateY(-1px);
    box-shadow: 0 3px 8px rgba(99, 102, 241, 0.08);
}

.stat-tile .num {
    font-size: 1.1rem;
    font-weight: 800;
    color: var(--accent);
    line-height: 1;
    letter-spacing: -0.02em;
}

.stat-tile .name {
    font-size: 0.58rem;
    color: var(--text-3);
    margin-top: 3px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.last-ing-block {
    margin-top: 8px;
    padding: 9px 10px;
    background: var(--surface);
    border: 1px solid var(--border-soft);
    border-radius: 8px;
    border-left: 3px solid var(--accent);
}

.last-ing-block .head {
    font-size: 0.58rem;
    font-weight: 700;
    color: var(--accent);
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin-bottom: 4px;
}

.last-ing-block .doc-name {
    font-size: 0.72rem;
    font-weight: 600;
    color: var(--text-1);
    margin-bottom: 3px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

.last-ing-block .date-row {
    font-size: 0.66rem;
    color: var(--text-2);
    margin-bottom: 3px;
}

.last-ing-block .storage-row {
    font-size: 0.64rem;
    color: var(--text-3);
    padding-top: 4px;
    margin-top: 4px;
    border-top: 1px dashed var(--border-soft);
}

.last-ing-block .storage-row strong {
    color: var(--accent);
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
}

.queue-item {
    background: var(--surface);
    border: 1px solid var(--border-soft);
    border-radius: 6px;
    padding: 5px 8px;
    margin: 3px 0;
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 0.7rem;
    color: var(--text-2);
}

.queue-item .icon { font-size: 0.82rem; flex-shrink: 0; }

.queue-item .name {
    flex: 1;
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    font-weight: 500;
    color: var(--text-1);
}

.queue-item .size {
    font-size: 0.6rem;
    color: var(--text-3);
    flex-shrink: 0;
    font-family: 'JetBrains Mono', monospace;
}

.indexed-card {
    background: var(--surface);
    border: 1px solid var(--border-soft);
    border-radius: 7px;
    padding: 6px 9px;
    margin: 3px 0;
    transition: all 0.15s ease;
}

.indexed-card:hover {
    border-color: var(--accent);
    box-shadow: 0 2px 6px rgba(99, 102, 241, 0.06);
}

.indexed-card .top {
    display: flex;
    align-items: center;
    gap: 6px;
    margin-bottom: 3px;
}

.indexed-card .top .check {
    color: var(--success);
    font-size: 0.78rem;
    flex-shrink: 0;
}

.indexed-card .name {
    flex: 1;
    min-width: 0;
    font-size: 0.7rem;
    font-weight: 600;
    color: var(--text-1);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

.indexed-card .meta {
    display: flex;
    gap: 4px;
    font-size: 0.58rem;
    color: var(--text-3);
    font-family: 'JetBrains Mono', monospace;
}

.indexed-card .meta span {
    background: var(--accent-bg);
    color: var(--accent);
    padding: 1px 5px;
    border-radius: 3px;
    font-weight: 600;
}

.danger-section {
    margin-top: 14px;
    padding-top: 10px;
    border-top: 1px dashed var(--border-soft);
}

[data-testid="stSidebar"] [data-testid="column"] {
    padding: 0 3px !important;
}

/* ========== MAIN AREA ========== */
.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

.main-top {
    padding: 18px 36px;
    border-bottom: 1px solid var(--border-soft);
    background: var(--bg);
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
    position: sticky;
    top: 0;
    z-index: 5;
}

.main-top-left {
    display: flex;
    align-items: center;
    gap: 16px;
    font-size: 1rem;
    color: var(--text-1);
    font-weight: 600;
}

.main-top-left .dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: var(--success);
    box-shadow: 0 0 0 3px var(--success-bg);
}

.main-top-left .sub {
    color: var(--text-3);
    font-weight: 500;
    margin-left: 4px;
}

.main-top-meta {
    display: flex;
    gap: 10px;
    font-size: 0.82rem;
    color: var(--text-2);
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
}

.main-top-meta span {
    background: var(--surface);
    border: 1px solid var(--border);
    padding: 6px 12px;
    border-radius: 8px;
}

.main-top-meta span strong {
    color: var(--accent);
    margin-right: 4px;
}

.chat-area {
    flex: 1;
    overflow-y: auto;
    padding: 24px 0 140px 0;
}

.chat-container {
    max-width: 820px;
    margin: 0 auto;
    padding: 0 32px;
    display: flex;
    flex-direction: column;
    gap: 22px;
}

/* ========== MESSAGES ========== */
.msg-row {
    display: flex;
    width: 100%;
    animation: msgFadeIn 0.3s ease;
}

@keyframes msgFadeIn {
    from { opacity: 0; transform: translateY(6px); }
    to { opacity: 1; transform: translateY(0); }
}

.msg-row.user { justify-content: flex-end; }
.msg-row.assistant { justify-content: flex-start; }

.user-col {
    display: flex;
    flex-direction: column;
    align-items: flex-end;
    max-width: 78%;
}

.msg-user {
    background: var(--user-bubble);
    color: var(--user-text);
    padding: 10px 16px;
    border-radius: 18px 18px 4px 18px;
    max-width: 100%;
    width: fit-content;
    font-size: 0.93rem;
    line-height: 1.55;
    white-space: pre-wrap;
    word-wrap: break-word;
    overflow-wrap: anywhere;
    box-shadow: var(--shadow);
}

.msg-timestamp {
    font-size: 0.66rem;
    color: var(--text-4);
    margin-top: 5px;
    font-family: 'JetBrains Mono', monospace;
}

.user-col .msg-timestamp { text-align: right; }

.msg-ai-wrap {
    display: flex;
    gap: 12px;
    max-width: 92%;
    width: 100%;
    align-items: flex-start;
}

.msg-ai-avatar {
    width: 32px;
    height: 32px;
    border-radius: 9px;
    background: var(--accent);
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-size: 0.78rem;
    font-weight: 800;
    flex-shrink: 0;
    box-shadow: 0 2px 6px rgba(99, 102, 241, 0.2);
}

.msg-ai-body {
    flex: 1;
    min-width: 0;
    display: flex;
    flex-direction: column;
    align-items: flex-start;
}

.msg-ai {
    background: var(--ai-bubble);
    color: var(--ai-text);
    padding: 14px 18px;
    border-radius: 4px 16px 16px 16px;
    font-size: 0.93rem;
    line-height: 1.65;
    border: 1px solid var(--border-soft);
    white-space: pre-wrap;
    word-wrap: break-word;
    box-shadow: var(--shadow);
    width: fit-content;
    max-width: 100%;
}

.msg-ai-body .msg-timestamp { text-align: left; }

.cursor-blink {
    display: inline-block;
    color: var(--accent);
    font-weight: 700;
    margin-left: 2px;
    animation: blink 1s steps(2) infinite;
}

@keyframes blink {
    50% { opacity: 0; }
}

.typing-bubble {
    background: var(--ai-bubble);
    border: 1px solid var(--border-soft);
    border-radius: 4px 16px 16px 16px;
    padding: 14px 18px;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    box-shadow: var(--shadow);
    width: fit-content;
}

.typing-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--accent);
    animation: typingDot 1.4s ease-in-out infinite;
}

.typing-dot:nth-child(2) { animation-delay: 0.2s; }
.typing-dot:nth-child(3) { animation-delay: 0.4s; }

@keyframes typingDot {
    0%, 60%, 100% { transform: scale(0.7); opacity: 0.5; }
    30% { transform: scale(1); opacity: 1; }
}

.welcome {
    max-width: 720px;
    margin: 80px auto 0;
    padding: 0 32px;
    text-align: center;
}

.welcome-logo {
    width: 64px;
    height: 64px;
    margin: 0 auto 24px;
    border-radius: 16px;
    background: var(--accent);
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-weight: 800;
    font-size: 1.6rem;
    box-shadow: 0 8px 24px rgba(99, 102, 241, 0.22);
}

.welcome-title {
    font-size: 2rem;
    font-weight: 700;
    color: var(--text-1);
    letter-spacing: -0.025em;
    margin-bottom: 10px;
    line-height: 1.15;
}

.welcome-sub {
    font-size: 0.95rem;
    color: var(--text-3);
    margin-bottom: 36px;
    line-height: 1.55;
    max-width: 480px;
    margin-left: auto;
    margin-right: auto;
}

section.main .stButton button[kind="secondary"] {
    background: var(--surface) !important;
    border: 1px solid var(--border-soft) !important;
    border-radius: 12px !important;
    padding: 16px 18px !important;
    text-align: left !important;
    color: var(--text-1) !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
    transition: all 0.15s ease !important;
    box-shadow: var(--shadow) !important;
    width: 100% !important;
    height: auto !important;
    line-height: 1.5 !important;
}

section.main .stButton button[kind="secondary"]:hover {
    border-color: var(--accent) !important;
    background: var(--accent-bg-2) !important;
    transform: translateY(-2px) !important;
    box-shadow: var(--shadow-lg) !important;
}

section.main [data-testid="stChatInput"] > div {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 16px !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.04) !important;
    transition: all 0.15s ease !important;
}

section.main [data-testid="stChatInput"] > div:focus-within {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 4px var(--accent-bg) !important;
}

section.main [data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: var(--text-1) !important;
    font-size: 0.95rem !important;
    padding: 14px 18px !important;
    min-height: 52px !important;
}

section.main [data-testid="stChatInput"] textarea::placeholder {
    color: var(--text-3) !important;
}

section.main [data-testid="stChatInput"] button {
    background: var(--accent) !important;
    color: white !important;
    border-radius: 10px !important;
}

section.main [data-testid="stChatInput"] button:hover {
    background: var(--accent-soft) !important;
}

[data-testid="stAlert"] {
    border-radius: 8px !important;
    border: 1px solid var(--border) !important;
    background: var(--surface-2) !important;
    color: var(--text-1) !important;
    padding: 6px 10px !important;
    font-size: 0.72rem !important;
}

[data-testid="stSidebar"] [data-testid="stAlert"] {
    padding: 5px 9px !important;
    font-size: 0.7rem !important;
}

.stCaption, [data-testid="stCaptionContainer"] {
    color: var(--text-3) !important;
    font-size: 0.66rem !important;
}

[data-testid="stSidebar"] .stMarkdown { margin-bottom: 0 !important; }
[data-testid="stSidebar"] hr { display: none !important; }

hr { display: none; }

::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
    background: var(--border);
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover { background: var(--text-4); }

@media (max-width: 900px) {
    [data-testid="stSidebar"] {
        min-width: 240px !important;
        max-width: 240px !important;
        width: 240px !important;
    }
    section[data-testid="stSidebar"] > div { width: 240px !important; }
    .chat-container { padding: 0 16px; }
    .main-top { padding: 14px 16px; }
    .welcome { padding: 0 16px; }
}
</style>

<script>
(function lockSidebarOpen() {
    function enforce() {
        const selectors = [
            '[data-testid="collapsedControl"]',
            '[data-testid="stSidebarCollapsedControl"]',
            '[data-testid="stSidebarCollapseButton"]',
            '[data-testid="stSidebarHeader"] button',
            'button[kind="header"]',
            'button[kind="headerNoPadding"]'
        ];
        selectors.forEach(sel => {
            document.querySelectorAll(sel).forEach(el => {
                el.style.display = 'none';
                el.style.visibility = 'hidden';
                el.style.pointerEvents = 'none';
            });
        });
        const sb = document.querySelector('[data-testid="stSidebar"]');
        if (sb) {
            sb.style.transform = 'translateX(0)';
            sb.style.visibility = 'visible';
            sb.style.marginLeft = '0';
            sb.setAttribute('aria-expanded', 'true');
        }
    }
    enforce();
    const obs = new MutationObserver(enforce);
    obs.observe(document.body, { childList: true, subtree: true, attributes: true });
})();
</script>
""", unsafe_allow_html=True)


# ============================================
# HELPERS
# ============================================

@st.cache_data(ttl=5)
def get_chunk_count():
    try:
        return repo.count()
    except Exception:
        return 0


@st.cache_data(ttl=5)
def count_visuals():
    try:
        if not os.path.exists(EXTRACTED_IMAGES_DIR):
            return 0
        count = 0
        for root, _, files in os.walk(EXTRACTED_IMAGES_DIR):
            count += sum(1 for f in files if f.lower().endswith((".png", ".jpg", ".jpeg")))
        return count
    except Exception:
        return 0


@st.cache_data(ttl=5)
def count_tables_global():
    try:
        results = repo.collection.get(where={"content_type": "table"}, limit=2000)
        return len(results.get("ids", []))
    except Exception:
        return 0


def get_total_storage_bytes():
    total = 0
    try:
        if os.path.exists(DATA_DIR):
            for f in os.listdir(DATA_DIR):
                fp = os.path.join(DATA_DIR, f)
                if os.path.isfile(fp):
                    total += os.path.getsize(fp)
    except Exception:
        pass
    return total


def get_last_ingestion_info():
    if not st.session_state.indexed_meta:
        return None, None, None
    last_doc = None
    last_dt = None
    for name, meta in st.session_state.indexed_meta.items():
        if "indexed_at_full" in meta:
            try:
                dt = datetime.strptime(meta["indexed_at_full"], "%Y-%m-%d %H:%M:%S")
                if last_dt is None or dt > last_dt:
                    last_dt = dt
                    last_doc = name
            except Exception:
                pass
    if last_doc and last_dt:
        return last_doc, last_dt.strftime("%d %b"), last_dt.strftime("%I:%M %p")
    return None, None, None


def list_docs():
    try:
        if not os.path.exists(DATA_DIR):
            return []
        supported = {".pdf", ".docx", ".txt", ".md", ".html", ".htm", ".pptx", ".xlsx", ".csv"}
        return sorted([
            f for f in os.listdir(DATA_DIR)
            if Path(f).suffix.lower() in supported
        ])
    except Exception:
        return []


def count_chunks_for_doc(filename):
    try:
        results = repo.collection.get(where={"source": filename}, limit=2000)
        return len(results.get("ids", []))
    except Exception:
        return 0


def count_per_type_for_doc(filename, content_type):
    try:
        results = repo.collection.get(
            where={"$and": [{"source": filename}, {"content_type": content_type}]},
            limit=2000
        )
        return len(results.get("ids", []))
    except Exception:
        return 0


def format_size(size_bytes):
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / 1024 / 1024:.1f} MB"


def save_uploads(files):
    if not files:
        return []
    os.makedirs(DATA_DIR, exist_ok=True)
    results = []
    for f in files:
        try:
            dest = os.path.join(DATA_DIR, f.name)
            if os.path.exists(dest) and os.path.getsize(dest) == f.size:
                results.append((f.name, f.size, "duplicate"))
                continue
            with open(dest, "wb") as out:
                out.write(f.getbuffer())
            results.append((f.name, f.size, "saved"))
        except Exception as e:
            results.append((getattr(f, "name", "?"), 0, f"error: {str(e)[:40]}"))
    return results


def run_ingestion(filenames, progress_cb=None):
    from src.parser import parse_file
    from src.captioner import caption_all_images
    from src.embedder import store_chunks
    try:
        from src.parser import reset_dedup_cache
        reset_dedup_cache()
    except Exception:
        pass

    all_text = []
    all_imgs = []
    for i, fname in enumerate(filenames):
        if progress_cb:
            progress_cb(f"Parsing {fname}...", (i / max(len(filenames), 1)) * 0.4)
        fp = os.path.join(DATA_DIR, fname)
        if not os.path.exists(fp):
            continue
        tc, imgs = parse_file(fp, fname)
        all_text.extend(tc)
        all_imgs.extend(imgs)

    if progress_cb:
        progress_cb("Captioning visuals...", 0.5)
    captioned = caption_all_images(all_imgs) if all_imgs else []

    if progress_cb:
        progress_cb("Embedding and storing...", 0.85)
    store_chunks(all_text, captioned)

    now = datetime.now()
    now_short = now.strftime("%d %b, %H:%M")
    now_full = now.strftime("%Y-%m-%d %H:%M:%S")
    for fname in filenames:
        st.session_state.indexed_meta[fname] = {
            "indexed_at": now_short,
            "indexed_at_full": now_full,
            "chunks": count_chunks_for_doc(fname),
            "images": count_per_type_for_doc(fname, "image") + count_per_type_for_doc(fname, "flowchart"),
            "tables": count_per_type_for_doc(fname, "table"),
        }
    get_chunk_count.clear()
    count_visuals.clear()
    count_tables_global.clear()


def clear_kb():
    try:
        repo.clear()
    except Exception as e:
        print(f"[clear_kb] repo.clear() failed: {e}")
    for folder in [DATA_DIR, EXTRACTED_IMAGES_DIR]:
        if os.path.exists(folder):
            for f in os.listdir(folder):
                try:
                    fp = os.path.join(folder, f)
                    if os.path.isfile(fp):
                        os.remove(fp)
                    elif os.path.isdir(fp):
                        shutil.rmtree(fp, ignore_errors=True)
                except Exception:
                    pass
    st.session_state.pending_files = []
    st.session_state.last_upload_id = None
    st.session_state.indexed_meta = {}
    get_chunk_count.clear()
    count_visuals.clear()
    count_tables_global.clear()


indexed_docs = list_docs()


# ============================================
# SIDEBAR  (URL section removed)
# ============================================
with st.sidebar:
    st.markdown("""
    <div class="brand">
        <div class="brand-mark">A</div>
        <div class="brand-info">
            <div class="brand-name">Atlas</div>
            <div class="brand-sub">Multimodal RAG</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("➕ New Chat", type="primary", key="new_chat_btn", use_container_width=True):
        clear_conversation(st.session_state)
        st.rerun()

    if st.session_state.messages:
        if st.button("🗑 Clear Conversation", type="secondary", key="clear_conv", use_container_width=True):
            clear_conversation(st.session_state)
            st.rerun()

    indexed_docs = list_docs()

    st.markdown(f'''
    <div class="sec-h">
        <span>📚 Knowledge Base</span>
        <span class="count">{len(indexed_docs)}</span>
    </div>
    ''', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload",
        type=["pdf", "docx", "txt", "md", "html", "htm", "pptx", "xlsx", "csv"],
        accept_multiple_files=True,
        key="uploader",
        label_visibility="collapsed"
    )

    if uploaded:
        upload_id = tuple(sorted((f.name, f.size) for f in uploaded))
        if upload_id != st.session_state.last_upload_id:
            st.session_state.last_upload_id = upload_id
            results = save_uploads(uploaded)
            for fname, size, status in results:
                if status == "saved":
                    if not any(p["name"] == fname for p in st.session_state.pending_files):
                        st.session_state.pending_files.append({"name": fname, "size": size})

    if st.session_state.pending_files:
        for item in st.session_state.pending_files:
            fname = item["name"]
            display = fname if len(fname) <= 18 else fname[:15] + "..."
            size_str = format_size(item["size"])
            st.markdown(f'''
            <div class="queue-item">
                <span class="icon">📄</span>
                <span class="name">{html_lib.escape(display)}</span>
                <span class="size">{size_str}</span>
            </div>
            ''', unsafe_allow_html=True)

        c1, c2 = st.columns([3, 1])
        with c1:
            if st.button("📥 Ingest", type="primary", key="ingest_btn", use_container_width=True):
                st.session_state.index_state = "processing"
                bar = st.progress(0.0, text="Starting...")

                def update_progress(msg, pct):
                    bar.progress(pct, text=msg)

                try:
                    filenames = [p["name"] for p in st.session_state.pending_files]
                    run_ingestion(filenames, progress_cb=update_progress)
                    bar.progress(1.0, text="Done!")
                    time.sleep(0.4)
                    st.session_state.pending_files = []
                    st.session_state.last_upload_id = None
                    st.session_state.ingest_status = "success"
                    st.session_state.index_state = "ready"
                except Exception as e:
                    st.session_state.ingest_status = f"error: {str(e)[:60]}"
                    st.session_state.index_state = "failed"
                st.rerun()
        with c2:
            if st.button("✕", type="secondary", key="cancel_queue", use_container_width=True):
                st.session_state.pending_files = []
                st.session_state.last_upload_id = None
                st.rerun()

    if st.session_state.ingest_status == "success":
        st.success("✅ Ingested")
        st.session_state.ingest_status = None
    elif st.session_state.ingest_status and str(st.session_state.ingest_status).startswith("error"):
        err = str(st.session_state.ingest_status).replace("error:", "", 1).strip()
        st.error(f"❌ {err}")
        st.session_state.ingest_status = None

    chunks_count = get_chunk_count()
    visuals_count = count_visuals()
    tables_count = count_tables_global()
    docs_count = len(indexed_docs)

    state = st.session_state.index_state or "ready"
    if docs_count == 0 and state == "ready":
        state_label = "Idle"
        state_class = "ready"
    else:
        state_label = {
            "ready": "Ready",
            "processing": "Building",
            "failed": "Failed",
        }.get(state, "Ready")
        state_class = state

    st.markdown(f'''
    <div class="stats-row" style="margin-top:10px;">
        <span class="label">📊 Status</span>
        <span class="index-pill">
            <span class="dot {state_class}"></span>
            {state_label}
        </span>
    </div>
    <div class="stats-grid">
        <div class="stat-tile">
            <div class="num">{docs_count}</div>
            <div class="name">Docs</div>
        </div>
        <div class="stat-tile">
            <div class="num">{chunks_count}</div>
            <div class="name">Chunks</div>
        </div>
        <div class="stat-tile">
            <div class="num">{visuals_count}</div>
            <div class="name">Visuals</div>
        </div>
        <div class="stat-tile">
            <div class="num">{tables_count}</div>
            <div class="name">Tables</div>
        </div>
    </div>
    ''', unsafe_allow_html=True)

    last_doc, last_date, last_time = get_last_ingestion_info()
    storage_bytes = get_total_storage_bytes()
    storage_str = format_size(storage_bytes)

    if last_doc or storage_bytes > 0:
        display_last = last_doc if last_doc else "—"
        if len(display_last) > 22:
            display_last = display_last[:19] + "..."

        if last_date and last_time:
            date_html = f'<div class="date-row">🕐 {last_date} · {last_time}</div>'
        else:
            date_html = '<div class="date-row" style="color:var(--text-3);">No ingestion yet</div>'

        st.markdown(f'''
        <div class="last-ing-block">
            <div class="head">📦 Last Ingestion</div>
            <div class="doc-name">{html_lib.escape(display_last)}</div>
            {date_html}
            <div class="storage-row">Storage: <strong>{storage_str}</strong></div>
        </div>
        ''', unsafe_allow_html=True)

    if indexed_docs:
        st.markdown(
            '<div class="sec-h" style="margin-top:10px;"><span>📂 Indexed</span></div>',
            unsafe_allow_html=True
        )

        for doc_name in indexed_docs[:5]:
            meta = st.session_state.indexed_meta.get(doc_name, {})
            if not meta:
                meta = {
                    "indexed_at": "—",
                    "chunks": count_chunks_for_doc(doc_name),
                    "images": count_per_type_for_doc(doc_name, "image") + count_per_type_for_doc(doc_name, "flowchart"),
                    "tables": count_per_type_for_doc(doc_name, "table"),
                }
                st.session_state.indexed_meta[doc_name] = meta

            display_name = doc_name if len(doc_name) <= 22 else doc_name[:19] + "..."
            meta_parts = []
            if meta.get("chunks", 0):
                meta_parts.append(f"<span>{meta['chunks']}c</span>")
            if meta.get("images", 0):
                meta_parts.append(f"<span>{meta['images']}i</span>")
            if meta.get("tables", 0):
                meta_parts.append(f"<span>{meta['tables']}t</span>")

            st.markdown(f'''
            <div class="indexed-card">
                <div class="top">
                    <span class="check">✓</span>
                    <span class="name">{html_lib.escape(display_name)}</span>
                </div>
                <div class="meta">{"".join(meta_parts)}</div>
            </div>
            ''', unsafe_allow_html=True)

        if len(indexed_docs) > 5:
            st.markdown(
                f'<div style="font-size:0.62rem;color:var(--text-3);text-align:center;margin-top:4px;">+{len(indexed_docs) - 5} more</div>',
                unsafe_allow_html=True
            )

    if indexed_docs or get_chunk_count() > 0:
        st.markdown('<div class="danger-section"></div>', unsafe_allow_html=True)
        if not st.session_state.confirm_clear_kb:
            if st.button("🗑 Clear Knowledge Base", type="secondary", key="clear_kb_btn", use_container_width=True):
                st.session_state.confirm_clear_kb = True
                st.rerun()
        else:
            st.warning("⚠ Delete all docs?")
            cc1, cc2 = st.columns(2)
            with cc1:
                if st.button("Yes", type="primary", key="confirm_yes", use_container_width=True):
                    clear_kb()
                    st.session_state.confirm_clear_kb = False
                    st.rerun()
            with cc2:
                if st.button("No", type="secondary", key="confirm_no", use_container_width=True):
                    st.session_state.confirm_clear_kb = False
                    st.rerun()


# ============================================
# TOP BAR
# ============================================
chunks_count = get_chunk_count()
visuals_count = count_visuals()
docs_count = len(indexed_docs)

st.markdown(f'''
<div class="main-top">
    <div class="main-top-left">
        <span class="dot"></span>
        <span>100% Local <span class="sub">· Private &amp; on-device</span></span>
    </div>
    <div class="main-top-meta">
        <span><strong>{docs_count}</strong>docs</span>
        <span><strong>{chunks_count}</strong>chunks</span>
        <span><strong>{visuals_count}</strong>visuals</span>
        <span><strong>LLM</strong>{html_lib.escape(LLM_MODEL)}</span>
    </div>
</div>
''', unsafe_allow_html=True)


# ============================================
# CHAT THREAD
# ============================================
def render_message(msg):
    role = msg["role"]
    content = html_lib.escape(msg["content"])
    time_str = msg.get("time", "")

    if role == "user":
        st.markdown(f'''
        <div class="msg-row user">
            <div class="user-col">
                <div class="msg-user">{content}</div>
                <div class="msg-timestamp">{time_str}</div>
            </div>
        </div>
        ''', unsafe_allow_html=True)
    else:
        st.markdown(f'''
        <div class="msg-row assistant">
            <div class="msg-ai-wrap">
                <div class="msg-ai-avatar">A</div>
                <div class="msg-ai-body">
                    <div class="msg-ai">{content}</div>
                    <div class="msg-timestamp">{time_str}</div>
                </div>
            </div>
        </div>
        ''', unsafe_allow_html=True)

        if SHOW_SOURCES_EXPANDER:
            sources = msg.get("sources", [])
            if sources:
                with st.expander(f"▼ View sources ({len(sources)})", expanded=False):
                    for i, src in enumerate(sources[:5]):
                        ct = src.get("content_type", "text")
                        fname = src.get("source", "document")
                        page = src.get("page", "—")
                        snippet = html_lib.escape(src.get("text", ""))
                        if len(snippet) > 240:
                            snippet = snippet[:240] + "..."
                        st.markdown(f"**[{i+1}] {ct.upper()}** · {fname} · p. {page}")
                        st.caption(snippet)


# ============================================
# CHAT INPUT
# ============================================
user_input = st.chat_input("Message Atlas...")

if not user_input and st.session_state.trigger_prompt:
    user_input = st.session_state.trigger_prompt
    st.session_state.trigger_prompt = None


if not user_input:
    if st.session_state.messages:
        st.markdown('<div class="chat-area"><div class="chat-container">', unsafe_allow_html=True)
        try:
            for msg in st.session_state.messages:
                render_message(msg)
        finally:
            st.markdown('</div></div>', unsafe_allow_html=True)
    else:
        st.markdown('''
        <div class="welcome">
            <div class="welcome-logo">A</div>
            <div class="welcome-title">What can I help you understand today?</div>
            <div class="welcome-sub">
                Ask anything about your indexed documents. I'll search through text, tables,
                images, and flowcharts to give you grounded answers.
            </div>
        </div>
        ''', unsafe_allow_html=True)

        examples = [
            "Summarize the key findings",
            "What concepts are explained in detail?",
            "Compare the approaches discussed",
            "What does the diagram on page 1 show?",
        ]

        ec_row1 = st.columns(2)
        ec_row2 = st.columns(2)

        with ec_row1[0]:
            if st.button(examples[0], key="ex_0_btn", use_container_width=True):
                st.session_state.trigger_prompt = examples[0]
                st.rerun()
        with ec_row1[1]:
            if st.button(examples[1], key="ex_1_btn", use_container_width=True):
                st.session_state.trigger_prompt = examples[1]
                st.rerun()
        with ec_row2[0]:
            if st.button(examples[2], key="ex_2_btn", use_container_width=True):
                st.session_state.trigger_prompt = examples[2]
                st.rerun()
        with ec_row2[1]:
            if st.button(examples[3], key="ex_3_btn", use_container_width=True):
                st.session_state.trigger_prompt = examples[3]
                st.rerun()


# ============================================
# PROCESS NEW MESSAGE
# ============================================
if user_input and user_input.strip():
    original_question = user_input.strip()

    user_msg = make_user_message(original_question)
    st.session_state.messages.append(user_msg)

    st.markdown('<div class="chat-area"><div class="chat-container">', unsafe_allow_html=True)
    try:
        for msg in st.session_state.messages:
            render_message(msg)
    finally:
        st.markdown('</div></div>', unsafe_allow_html=True)

    history_before_this = st.session_state.messages[:-1]

    t_rewrite_start = time.time()
    try:
        retrieval_query = rewrite_question(original_question, history_before_this)
    except Exception:
        retrieval_query = original_question
    rewrite_time = time.time() - t_rewrite_start

    history_text = format_history_for_prompt(history_before_this)

    overall_start = time.time()

    if ENABLE_STREAMING:
        placeholder = st.empty()

        placeholder.markdown('''
        <div class="msg-row assistant">
            <div class="msg-ai-wrap">
                <div class="msg-ai-avatar">A</div>
                <div class="msg-ai-body">
                    <div class="typing-bubble">
                        <div class="typing-dot"></div>
                        <div class="typing-dot"></div>
                        <div class="typing-dot"></div>
                    </div>
                </div>
            </div>
        </div>
        ''', unsafe_allow_html=True)

        accumulated = ""
        final_sources = []
        error_text = None

        try:
            for kind, payload in ask_stream(retrieval_query, conversation_history_text=history_text):
                if kind == "token":
                    accumulated += payload
                    safe = html_lib.escape(accumulated)
                    placeholder.markdown(f'''
                    <div class="msg-row assistant">
                        <div class="msg-ai-wrap">
                            <div class="msg-ai-avatar">A</div>
                            <div class="msg-ai-body">
                                <div class="msg-ai">{safe}<span class="cursor-blink">▎</span></div>
                            </div>
                        </div>
                    </div>
                    ''', unsafe_allow_html=True)
                elif kind == "sources":
                    final_sources = payload
                elif kind == "error":
                    error_text = payload
                    break
        except Exception as e:
            error_text = str(e)

        elapsed = time.time() - overall_start

        if error_text:
            final_content = f"Sorry, I ran into an error: {error_text}"
        elif not accumulated:
            final_content = "(empty response)"
        else:
            final_content = accumulated

        placeholder.empty()

        assistant_msg = make_assistant_message(
            content=final_content,
            sources=final_sources,
            confidence="",
            generation_time=elapsed,
        )
        st.session_state.messages.append(assistant_msg)

        if ENABLE_PROFILING:
            total = rewrite_time + elapsed
            print(
                f"   [Pipeline] rewrite={rewrite_time:.2f}s "
                f"answer={elapsed:.2f}s "
                f"total={total:.2f}s"
            )

        st.rerun()

    else:
        elapsed = 0.0
        with st.spinner(""):
            try:
                start = time.time()
                answer, sources = ask(retrieval_query, conversation_history_text=history_text)
                elapsed = time.time() - start

                assistant_msg = make_assistant_message(
                    content=answer,
                    sources=sources,
                    confidence="",
                    generation_time=elapsed,
                )
                st.session_state.messages.append(assistant_msg)
            except Exception as e:
                error_msg = make_assistant_message(
                    content=f"Sorry, I ran into an error: {e}",
                    sources=[],
                    confidence="",
                    generation_time=0.0,
                )
                st.session_state.messages.append(error_msg)

        if ENABLE_PROFILING:
            print(f"   [Pipeline] rewrite={rewrite_time:.2f}s total={elapsed:.2f}s")

        st.rerun()


if st.session_state.messages:
    st.markdown('''
    <script>
        setTimeout(function() {
            window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });
        }, 80);
    </script>
    ''', unsafe_allow_html=True)