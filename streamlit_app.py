#!/usr/bin/env python3
"""
RAG, AI Agents & Generative AI Course Assistant - Streamlit App
Cyber Diogo's AI-powered companion for your Udemy course
"""

import os
import re
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from streamlit_extras.buy_me_a_coffee import button

# LangSmith tracing
try:
    from langsmith.wrappers import wrap_openai
    from langsmith import traceable
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False
    print("Warning: LangSmith not available - install langsmith for tracing")

# Import our tracing config
try:
    from tracing_config import get_langsmith_config, is_langsmith_configured
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    print("Warning: Tracing config not available")

# Load environment variables from .env file in the current directory
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path, override=True)
print(f"Info: Loading .env from {env_path}")

# Debug: Check what API key is loaded
api_key = os.getenv("OPENAI_API_KEY")

# Configure OpenAI client with LangSmith tracing
if LANGSMITH_AVAILABLE and is_langsmith_configured():
    # Wrap the client for automatic tracing
    client = wrap_openai(OpenAI(api_key=api_key))
    print("Success: LangSmith tracing enabled for OpenAI client")
else:
    # Use regular client without tracing
    client = OpenAI(api_key=api_key)
    if LANGSMITH_AVAILABLE:
        print("Warning: LangSmith available but not configured - check environment variables")
    else:
        print("Info: LangSmith not available - using regular OpenAI client")

# Page configuration
st.set_page_config(
    page_title="RAG & AI Agents Course Assistant",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* optional: tighten section gaps */
    section.main > div { padding-top: 0.25rem; }
    .main-header {
        /* Main title - largest and most prominent */
        font-size: clamp(2.5rem, 6vw, 3rem) !important;
        font-weight: 900;
        color: #0074FF;
        text-align: center;
        margin-bottom: 1.5rem;
        word-wrap: break-word;
        overflow-wrap: break-word;
        max-width: 100%;
        line-height: 1.1;
        padding: 0 0.5rem;
        white-space: normal;
        overflow: visible;
        letter-spacing: -0.02em;
    }
    
    /* Tagline styling */
    .tagline {
        font-size: clamp(1.1rem, 2.5vw, 1.3rem) !important;
        font-weight: 400;
        color: #666666;
        text-align: center;
        margin-bottom: 2.5rem;
        line-height: 1.4;
    }
    
    /* Welcome section styling */
    .welcome-section {
        margin-bottom: 2rem;
    }
    
    .welcome-title {
        font-size: clamp(1.6rem, 3.5vw, 2rem) !important;
        font-weight: 700;
        color: #373435;
        margin-bottom: 1rem;
        line-height: 1.3;
    }
    
    .welcome-intro {
        font-size: clamp(1rem, 2vw, 1.1rem) !important;
        font-weight: 400;
        color: #555555;
        margin-bottom: 1.5rem;
        line-height: 1.5;
    }
    
    .capabilities-list {
        font-size: clamp(0.95rem, 1.8vw, 1rem) !important;
        line-height: 1.6;
        margin-bottom: 1.5rem;
        color: #373435;
    }
    
    .capabilities-list strong {
        font-weight: 600;
        color: #373435;
    }
    
    .capabilities-list ul {
        margin: 0;
        padding-left: 1.5rem;
    }
    
    .capabilities-list li {
        margin-bottom: 0.5rem;
        padding-left: 0.5rem;
    }
    
    .call-to-action {
        font-size: clamp(1rem, 2vw, 1.1rem) !important;
        font-weight: 500;
        color: #000000;
        font-style: italic;
        margin-top: 1.5rem;
    }
    
    /* ChatGPT-like message styling */
    [data-testid="stChatMessageContainer"] {
        max-width: 760px;
        margin: 0 auto;
    }
    [data-testid="stChatMessage"] {
        align-items: flex-start;
        gap: 0.75rem;
    }
    [data-testid="stChatMessage"] [data-testid="stChatMessageMessage"] {
        font-size: 0.95rem;
        line-height: 1.6;
    }
    [data-testid="stChatMessageUser"] {
        flex-direction: row-reverse;
        justify-content: flex-end;
    }
    [data-testid="stChatMessageUser"] [data-testid="stChatMessageAvatar"] {
        order: 2;
        margin-left: 0.75rem;
        margin-right: 0;
    }
    [data-testid="stChatMessageUser"] [data-testid="stChatMessageMessage"] {
        order: 1;
        background: linear-gradient(135deg, #dbeafe, #bfdbfe);
        color: #0f172a;
        border-radius: 18px;
        padding: 12px 16px;
        box-shadow: 0 8px 20px rgba(59, 130, 246, 0.18);
        border: 1px solid rgba(59, 130, 246, 0.35);
        margin-left: auto;
        margin-right: 0;
    }
    [data-testid="stChatMessageAssistant"] [data-testid="stChatMessageMessage"] {
        background: #ffffff;
        border-radius: 18px;
        padding: 12px 16px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
        margin-right: auto;
        margin-left: 0.75rem;
    }
    [data-testid="stChatMessage"] ul,
    [data-testid="stChatMessage"] ol {
        margin: 0.25rem 0 0.75rem 1.25rem;
        padding-left: 0.75rem;
    }
    [data-testid="stChatMessage"] li {
        margin-bottom: 0.35rem;
    }
    [data-testid="stChatMessage"] pre {
        background: #0f172a;
        color: #f8fafc;
        padding: 1rem 1.25rem;
        border-radius: 14px;
        margin: 0.75rem 0;
        box-shadow: inset 0 0 0 1px rgba(148, 163, 184, 0.24);
        overflow-x: auto;
    }
    [data-testid="stChatMessage"] code {
        font-family: 'Fira Code', 'Source Code Pro', monospace;
        font-size: 0.95rem;
    }
    
    /* Quick action buttons */
    .quick-action-btn {
        background-color: #007bff;
        color: white;
        border: none;
        border-radius: 20px;
        padding: 8px 16px;
        margin: 4px;
        cursor: pointer;
        font-size: 14px;
        transition: all 0.3s ease;
    }
    
    .quick-action-btn:hover {
        background-color: #0056b3;
        transform: translateY(-1px);
    }
    
    /* Better button styling */
    .stButton > button {
        border-radius: 12px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(0, 116, 255, 0.2);
        background: linear-gradient(135deg, #0074FF, #0056CC) !important;
        color: white !important;
        border: none !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 116, 255, 0.35);
        background: linear-gradient(135deg, #0056CC, #003D99) !important;
    }
    
    /* Chat input styling */
    [data-testid="stChatInputTextArea"] textarea {
        border-radius: 16px !important;
        border: 1px solid #d1d5db !important;
        padding: 0.75rem 1rem !important;
        font-size: 0.95rem !important;
        background: #ffffff !important;
        color: #111827 !important;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.06) !important;
    }
    [data-testid="stChatInputTextArea"] textarea:focus {
        border-color: #2563eb !important;
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.15) !important;
        outline: none !important;
    }
    [data-testid="stChatInputSubmitButton"] button {
        background: linear-gradient(135deg, #0074FF, #0056CC) !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        border-radius: 999px !important;
        padding: 0.6rem 1.4rem !important;
        box-shadow: 0 12px 24px rgba(59, 130, 246, 0.35) !important;
        border: none !important;
    }
    [data-testid="stChatInputSubmitButton"] button:hover {
        transform: translateY(-1px);
        box-shadow: 0 16px 32px rgba(59, 130, 246, 0.4) !important;
    }


</style>
""", unsafe_allow_html=True)

# Initialize session state
if "last_response_id" not in st.session_state:
    st.session_state.last_response_id = None
if "vector_store_id" not in st.session_state:
    st.session_state.vector_store_id = None


# Load vector store metadata
def load_vector_store():
    """Load vector store ID from .env file first, then fall back to Streamlit secrets"""
    try:
        # First try to load from .env file (for local development)
        vector_store_id = os.getenv("VECTOR_STORE_ID")
        
        if not vector_store_id:
            # Fall back to Streamlit secrets (for production deployment)
            try:
                vector_store_id = st.secrets.get("VECTOR_STORE_ID")
            except Exception:
                pass
        
        if not vector_store_id:
            st.error("❌ Vector store ID not found. Please check your .env file or .streamlit/secrets.toml file.")
            return None
            
        return vector_store_id
    except Exception as e:
        st.error(f"❌ Error loading vector store ID: {str(e)}")
        return None

# Get API key from .env file
def get_api_key():
    """Get API key from .env file."""
    return os.getenv("OPENAI_API_KEY")


# System instructions
SYSTEM_INSTRUCTIONS = """
You are Cyber Diogo, the assistant for the Udemy course "RAG, AI Agents and Generative AI with Python and OpenAI".
Be concise, direct, and practical. Use active voice. No fluff.

Primary objective
- Answer questions about the course content and code using the attached Vector Store (transcripts, notebooks, scripts).
- Prefer retrieved facts over memory. If the files don't cover it, say so.

Retrieval & citations
- Always use File Search first.
- Ground every substantive answer in retrieved snippets.
- If nothing relevant is found, say: "I don't see this in the course files." Then suggest the most relevant module(s) the learner should review.
- Never include source citations or reference labels in the final answer text.

Answer style
- Keep outputs scannable: short paragraphs, bullet steps, compact runnable code samples when needed.
- When explaining "how to build X", outline the pipeline stages (ingest → retrieve → generate → evaluate) before diving into code.
- Close each reply with a friendly follow-up question the learner might ask next.
- Stay approachable, encouraging, and human.

Boundaries
- Don't invent references, credentials, metrics, or file names.
- If the topic is outside RAG/agents/this curriculum, acknowledge the gap and offer a high-level pointer or ask for clarification.

Context: Course map & typical intents
- Section 1: Orientation—promo, overview, resource downloads, course assistant setup, instructor background, and request channels for 2026 updates.
- Section 2: Python for RAG & AI—functions, control flow, classes, exercises, and mini-games that refresh foundational skills.
- Part A (Sections 3–11): Flowise RAG builds—planning, loaders, chunking, embeddings, vector stores, prompt engineering (system, temperature/top_p, persona), quizzes, and the first RAG capstone.
- Part B (Sections 13–16): OpenAI API pipelines—text & image endpoints, Responses API, GenAI customer acquisition project, file search assistant in Streamlit, configuration, debugging, and deployment.
- Part C (Sections 17–20): Unstructured & multimodal RAG—LangChain ingestion, FAISS indexing, Whisper transcription, CLIP embeddings, contrastive learning, video/audio pipelines, and Starbucks financial capstone.
- Part D (Sections 21–24): Advanced retrieval—LightRAG knowledge graphs, agentic RAG state machines, multi-agent collaboration, RAGAS metrics, and evaluation workflows.
- Part E (Sections 25–29): AI agents—CrewAI tooling, OpenAI Swarm orchestration, news-fetching crews, The Psychiatrist capstone, and human-in-the-loop guardrails.
- Part F (Sections 30–35): Advanced generative AI—reasoning models (o-series), image generation pipelines, GPT fine-tuning end-to-end, MCP integrations, and closing surveys.
- Appendix (Sections 36–45): Python crash course, essentials, intermediate/advanced drills, OOP projects, capstone games, book reviews, and next steps guidance.

If the learner references a lecture/section by name/number, search for files with that stem and tailor the answer.
Never invent lecture numbers or titles—they change over time.
If the answer isn’t in the corpus, say so clearly.
"""

INITIAL_ASSISTANT_MESSAGE = """
I'm Cyber Diogo, your RAG & AI Agents wingman! 🤖
Ask me anything about the course.
"""

# Formatting helpers ---------------------------------------------------------

CODE_LINE_PREFIXES = (
    "import ", "from ", "def ", "class ", "return", "yield",
    "for ", "while ", "if ", "elif ", "else:", "try:", "except",
    "with ", "@", "async ", "await ", "raise ", "assert ", "print(",
)
ASSIGNMENT_PATTERN = re.compile(r"^[A-Za-z_][\w\d_]*(\[[^\]]+\])?\s*=\s*.+")
ORDERED_LIST_PATTERN = re.compile(r"^\d+[\).\s]")


def _looks_like_code_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if stripped.startswith(("#", "- ", "* ", ">")):
        return False
    if ORDERED_LIST_PATTERN.match(stripped):
        return False
    lowered = stripped.lower()
    if lowered.startswith(("note:", "tip:", "warning:", "source:")):
        return False
    if ASSIGNMENT_PATTERN.match(stripped):
        return True
    if any(stripped.startswith(prefix) for prefix in CODE_LINE_PREFIXES):
        return True
    if stripped.endswith(":") and not stripped.endswith(("..", "...")):
        return True
    if "(" in stripped and ")" in stripped and any(op in stripped for op in "=+-*/"):
        return True
    return False


def _is_code_block(lines):
    meaningful = [line for line in lines if line.strip()]
    if len(meaningful) < 2:
        return False
    score = sum(1 for line in meaningful if _looks_like_code_line(line))
    return score >= max(2, len(meaningful) * 0.6)


def _normalize_code_lines(lines):
    normalized = []
    for line in lines:
        stripped = line.strip()
        if stripped.lower().startswith("python "):
            stripped = stripped.split(" ", 1)[1] if " " in stripped else ""
        normalized.append(stripped if stripped else "")
    return normalized


def _format_plain_segment(segment: str) -> str:
    paragraphs = [para for para in re.split(r"\n\s*\n", segment) if para.strip()]
    formatted_parts = []
    for para in paragraphs:
        lines = para.splitlines()
        if _is_code_block(lines):
            code_lines = _normalize_code_lines(lines)
            # Avoid empty code block
            code_text = "\n".join(code_lines).strip("\n")
            if code_text:
                formatted_parts.append(f"```python\n{code_text}\n```")
        else:
            formatted_parts.append(para.strip())
    return "\n\n".join(formatted_parts)


def format_for_chat(content: str) -> str:
    if not content:
        return content

    pieces = []
    pattern = re.compile(r"```.*?```", re.DOTALL)
    last_idx = 0

    for match in pattern.finditer(content):
        pre_segment = content[last_idx:match.start()]
        if pre_segment.strip():
            pieces.append(_format_plain_segment(pre_segment))
        pieces.append(match.group().strip("\n"))
        last_idx = match.end()

    tail = content[last_idx:]
    if tail.strip():
        pieces.append(_format_plain_segment(tail))

    formatted = "\n\n".join(piece.strip() for piece in pieces if piece).strip()
    return formatted or content


# Ensure chat history starts with the assistant's welcome message
if "messages" not in st.session_state or not st.session_state.messages:
    st.session_state.messages = [{
        "role": "assistant",
        "content": format_for_chat(INITIAL_ASSISTANT_MESSAGE.strip())
    }]

# OpenAI client setup
@traceable if LANGSMITH_AVAILABLE else lambda x: x
def ask_bot(user_question: str, verbosity: str = "low"):
    common_kwargs = {
        "model": "gpt-5-nano",
        "tools": [{"type": "file_search", "vector_store_ids": [st.session_state.vector_store_id]}],
        "text": {"verbosity": verbosity}
        }

    if st.session_state.last_response_id:
        # Continue the same conversation on the server
        resp = client.responses.create(
            previous_response_id=st.session_state.last_response_id,
            input=[{"role": "user", "content": user_question}],
            **common_kwargs,
        )
    else:
        # First turn: seed with system + initial assistant message
        resp = client.responses.create(
            input=[
                {"role": "system", "content": SYSTEM_INSTRUCTIONS},
                {"role": "assistant", "content": INITIAL_ASSISTANT_MESSAGE},
                {"role": "user", "content": user_question},
            ],
            **common_kwargs,
        )

    st.session_state.last_response_id = resp.id
    return resp.output_text

def reset_conversation():
    """Reset the conversation history."""
    st.session_state.last_response_id = None
    st.session_state.messages = [{
        "role": "assistant",
        "content": format_for_chat(INITIAL_ASSISTANT_MESSAGE.strip())
    }]
    st.rerun()

# Main app
def main():
    # Add LangSmith tracing for app startup
    if LANGSMITH_AVAILABLE and is_langsmith_configured():
        print("Info: App started - LangSmith tracing active")
        config = get_langsmith_config()
        print(f"Project info: {config['project']} (environment: {config['environment']})")
    elif LANGSMITH_AVAILABLE:
        print("Info: LangSmith available but not configured")
    else:
        print("Info: LangSmith not available")
    
    # Header
    st.markdown('<h1 class="main-header">🤖 RAG, AI Agents & Generative AI Course Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p class="tagline">Your AI co-pilot for Retrieval-Augmented Generation, agent workflows, and OpenAI-powered applications. 🚀</p>', unsafe_allow_html=True)
    
    # Introduction section
    with st.expander("📚 What I Can Help You With", expanded=False):
        st.markdown("""
        **🎯 My Capabilities:**
        - **Implementation Boosts**: Provide runnable snippets for ingestion jobs, vector store prep, LangChain workflows, CrewAI/Swarm agents, and Streamlit deployment.
        - **Concept Clarity**: Explain chunking heuristics, embedding choices, retrieval quality levers, prompt engineering, guardrails, and evaluation frameworks in plain language.
        - **File References**: Surface the exact lecture notes, notebooks, datasets, or helper scripts you need, with context and citations.
        - **Architecture Advice**: Compare Flowise vs. native OpenAI pipelines, hybrid search strategies, agent routing patterns, and monitoring approaches.
        - **Capstone Coaching**: Walk you through projects like the customer acquisition copilot, multimodal Starbucks analyst, and AI product manager crews.
        
        **📖 Course Topics I Cover:**
        - **Section 1**: Orientation—promo, course overview, materials, assistant access, instructor intro, and update requests.
        - **Section 2**: Python for RAG & AI—functions, exercises, classes, and mini-games to sharpen automation skills.
        - **Part A (Sections 3–11)**: RAG fundamentals with Flowise—planning, loaders, chunking, embeddings, vector stores, prompt engineering, challenges, and the first RAG capstone.
        - **Part B (Sections 13–16)**: RAG with OpenAI API—Responses API, text & image workflows, GenAI customer acquisition project, file search build in Streamlit, and deployment pipeline.
        - **Part C (Sections 17–20)**: Unstructured & multimodal data—LangChain ingestion, FAISS, Whisper, CLIP, contrastive learning, and the Starbucks multimodal project.
        - **Part D (Sections 21–24)**: Advanced retrieval—LightRAG knowledge graphs, agentic RAG orchestration, state management, and evaluation with RAGAS metrics.
        - **Part E (Sections 25–29)**: AI agents—CrewAI crews, OpenAI Swarm orchestration, pricing research agents, The Psychiatrist capstone, and human-in-the-loop patterns.
        - **Part F (Sections 30–35)**: Advanced generative AI—reasoning models (O-series), image workflows, GPT fine-tuning, MCP integrations, and course wrap-up.
        - **Appendix (Sections 36–45)**: Python crash course, fundamentals, intermediate challenges, OOP deep dives, capstone games, and next-step guidance.

        **❌ What I Cannot Do:**
        - Run code or execute scripts (I provide code, you run it)
        - Access external websites or real-time data
        - Remember conversations between sessions
        - Provide financial or medical advice
        - Solve problems outside the course scope
        """)
    

    
    # Load vector store
    if not st.session_state.vector_store_id:
        st.session_state.vector_store_id = load_vector_store()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # LangSmith tracing status indicator
        # if LANGSMITH_AVAILABLE and is_langsmith_configured():
        #     st.success("🔍 AI Assistant is here for you", icon="✅") 

        # Reset button
        if st.button("🔄 Reset Conversation", type="secondary"):
            reset_conversation()
        
        # Feedback section
        st.markdown("---")
        st.markdown("### 💬 Share Your Thoughts")
        st.markdown("Help me improve!")
        
        # Feedback button - direct link to Typeform
        st.markdown(f"""
        <a href="https://6yoersztgja.typeform.com/to/A8OXdhvY" target="_blank" style="text-decoration: none;">
            <button style="width: 100%; background-color: #0074FF; color: white; border: none; border-radius: 8px; padding: 0.5rem 1rem; font-size: 14px; font-weight: 500; cursor: pointer; transition: all 0.3s ease; display: flex; align-items: center; justify-content: center; gap: 8px;">
                📝 Give Feedback
            </button>
        </a>
        """, unsafe_allow_html=True)

        # Course link
        st.markdown("---")
        st.markdown("### 📚 Access the Course")
        st.markdown("Ready to master RAG and AI agent design?")
        
        # Course button - direct link
        st.markdown(f"""
        <a href="https://www.udemy.com/course/generative-ai-rag/" target="_blank" style="text-decoration: none;">
            <button style="width: 100%; background-color: #0074FF; color: white; border: none; border-radius: 8px; padding: 0.5rem 1rem; font-size: 14px; font-weight: 500; cursor: pointer; transition: all 0.3s ease; display: flex; align-items: center; justify-content: center; gap: 8px;">
                🚀 Go to Course
            </button>
        </a>
        """, unsafe_allow_html=True)
        
        st.markdown("---")

        # Support section
        st.markdown("### 💝 Support This App")
        st.markdown("Help keep this RAG & AI Agents Course Assistant free and running!")
        
        
        # Buy Me a Coffee button
        button(
            username="diogoalvesx",  # Replace with your actual username
            floating=True,
            text="Buy me a coffee",
            emoji="☕",
            bg_color="#0074FF",  # Your brand blue
            font_color="#FFFFFF",  # White text
            coffee_color="#FFFFFF",  # White coffee icon
            width=300
        )
        

    
    # Conversation history (ChatGPT-style)
    for message in st.session_state.messages:
        display_text = (
            format_for_chat(message["content"])
            if message["role"] == "assistant"
            else message["content"]
        )
        if message["role"] == "assistant" and display_text != message["content"]:
            message["content"] = display_text
        with st.chat_message(message["role"]):
            st.markdown(display_text)

    # Chat input aligned with Streamlit's native component
    prompt = st.chat_input("Ask me anything about the course...")
    if prompt:
        # echo user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # assistant thinking + response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                response = ask_bot(prompt)
            formatted_response = format_for_chat(response)
            st.markdown(formatted_response)

        st.session_state.messages.append({"role": "assistant", "content": formatted_response})

if __name__ == "__main__":
    main()
