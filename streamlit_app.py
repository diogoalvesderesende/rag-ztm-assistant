#!/usr/bin/env python3
"""
RAG, AI Agents & Generative AI Course Assistant - Streamlit App
Cyber Diogo's AI-powered companion for your Udemy course
"""

import streamlit as st
import json
import os
from pathlib import Path
from typing import Optional
import time
from datetime import datetime
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
    
    /* Enhanced message styling */
    .user-message {
        background: linear-gradient(135deg, rgba(230, 240, 255, 0.8), rgba(200, 220, 255, 0.6));
        border-radius: 18px;
        padding: 14px 18px;
        margin: 8px 0;
        border-left: 3px solid #0074FF;
        text-align: right;
        margin-left: 15%;
        box-shadow: 0 2px 8px rgba(0, 116, 255, 0.15);
        color: #1a1a1a;
        font-size: 15px;
        line-height: 1.5;
    }
    
    .assistant-message {
        background: linear-gradient(135deg, rgba(248, 248, 250, 0.95), rgba(240, 242, 250, 0.9));
        border-radius: 18px;
        padding: 14px 18px;
        margin: 8px 0;
        border-left: 3px solid #705FFE;
        text-align: left;
        margin-right: 15%;
        box-shadow: 0 2px 8px rgba(112, 95, 254, 0.12);
        color: #1a1a1a;
        border: 1px solid rgba(112, 95, 254, 0.15);
        font-size: 15px;
        line-height: 1.6;
    }
    
    /* Code block styling */
    .code-block {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 12px;
        font-family: 'Courier New', monospace;
        margin: 8px 0;
        position: relative;
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
    
    /* Text input styling */
    .stTextArea textarea {
        border-radius: 12px;
        border: 2px solid #e0e0e0;
        padding: 12px;
        font-size: 15px;
        transition: all 0.3s ease;
    }
    
    .stTextArea textarea:focus {
        border-color: #0074FF;
        box-shadow: 0 0 0 2px rgba(0, 116, 255, 0.1);
        outline: none;
    }
    
    /* Form styling */
    [data-testid="stForm"] {
        background: transparent;
        border: none;
        padding: 0;
    }
    
    /* Override any Streamlit default button colors */
    button[data-baseweb="button"] {
        background-color: #0074FF !important;
        color: white !important;
    }
    
    /* Improved text input */
    .stTextArea > div > div > textarea {
        border-radius: 15px;
        border: 3px solid #E0E0E0;
        transition: border-color 0.0s ease;
    }
    
    .stTextArea > div > div > textarea:focus {
        border-color: #0074FF;
        box-shadow: 0 0 0 2px rgba(0, 116, 255, 0.2);
    }
    
    /* Kill ALL borders and shadows around the text area wrapper */
    .stTextArea, 
    .stTextArea > div, 
    .stTextArea > div > div {
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
        background: transparent !important; /* or white if you prefer */
    }
            
    .stTextArea > div > div > textarea {
        font-family: 'Montserrat', sans-serif !important;
        color: #373435 !important;  /* dark gray body text */
        background-color: #FFFFFF !important; /* clean white input */
        border-radius: 15px !important;
        border: 2px solid #E0E0E0 !important; /* soft gray border */
    }
    .stTextArea > div > div > textarea:focus {
        border-color: #0074FF !important;  /* brand blue focus */
        box-shadow: 0 0 0 2px rgba(0, 116, 255, 0.2) !important;
    }

    

</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
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

def display_message(role, content, timestamp=None):
    """Display a message with proper styling and metadata"""
    def clean(txt: str) -> str:
        # remove both raw and escaped <br> variants
        replacements = [
            '<br>', '<br/>', '<br />',
            '&lt;br&gt;', '&lt;br/&gt;', '&lt;br /&gt;'
        ]
        for r in replacements:
            txt = txt.replace(r, '')
        return txt.strip()

    safe = clean(content)

    if role == "user":
        st.markdown(f"""
        <div class="user-message">
            <strong>You</strong>{f' <small style="color:#666;">{timestamp}</small>' if timestamp else ''}
            <br>{safe}
        </div>
        """, unsafe_allow_html=True)
    else:
        # Assistant content can include HTML formatting—keep as is
        st.markdown(f"""
        <div class="assistant-message">
            <strong>Assistant</strong>{f' <small style="color:#666;">{timestamp}</small>' if timestamp else ''}
            <br>{content}
        </div>
        """, unsafe_allow_html=True)



def display_code_snippet(code, language="python", filename=None):
    """Display code with syntax highlighting and copy button"""
    st.markdown(f"""
    <div class="code-block">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <strong>{filename or 'Code Snippet'}</strong>
            <button onclick="navigator.clipboard.writeText(`{code}`)" class="quick-action-btn">
                📋 Copy
            </button>
        </div>
        <pre><code class="language-{language}">{code}</code></pre>
    </div>
    """, unsafe_allow_html=True)

def process_user_input(user_input):
    """Process user input and generate response"""
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
        
    # Get bot response
    with st.spinner("🤔 Thinking..."):
        response = ask_bot(user_input)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Rerun to show new messages
    st.rerun()

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
Ask me anything about the course—from building ingestion pipelines and tuning retrieval to orchestrating OpenAI Swarm agents or deploying the capstones.
Try: "How do we evaluate our retriever with RAGAS?", 
"Show the ingestion script for PDFs", or
 "When should I use Swarm over a CrewAI agent?"

"""

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
    print(resp.output_text)
    return resp.output_text

def reset_conversation():
    """Reset the conversation history."""
    st.session_state.last_response_id = None
    st.session_state.messages = []
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
        

    
    # Welcome message and capabilities - always visible
    st.markdown('<div class="welcome-section">', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Main chat area - show messages when they exist
    if st.session_state.messages:
        # Add some spacing before chat messages
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Display chat messages with enhanced styling
        for i, message in enumerate(st.session_state.messages):
            # Add timestamp for recent messages
            timestamp = None
            if i == len(st.session_state.messages) - 1:
                timestamp = datetime.now().strftime("%H:%M")
            
            display_message(message["role"], message["content"], timestamp)
    
    # Enhanced chat input at the bottom
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        with st.form("chat_form", clear_on_submit=True):
            user_input = st.text_area(
                "Your question:",
                placeholder="Enter your message here....",
                key="user_input",
                height=80,
                label_visibility="collapsed"
            )
            
            submitted = st.form_submit_button("🚀 Send", use_container_width=True, type="primary", help="Send your message")
            if submitted and user_input.strip():
                process_user_input(user_input.strip())

if __name__ == "__main__":
    main()
