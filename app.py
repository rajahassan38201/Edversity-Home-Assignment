"""
Streamlit UI for the LearnForge Support Assistant.
Features:
- Chat-style interface with message history
- Per-message query stats (Confidence, Intent, Action, etc.)
- Bold green highlighted citations for clear grounding
- "Thinking..." indicator shown after user message
- Color-coded citation badges by source type
- Visual differentiation: escalations (red), clarifications (blue)
"""
import re
import streamlit as st
from src.graph import run_agent

# ---------- Page config ----------
st.set_page_config(
    page_title="LearnForge Support Assistant",
    page_icon="🎓",
    layout="wide",
)

# ---------- Custom CSS ----------
st.markdown(
    """
<style>
    .main-header { font-size: 2rem; font-weight: 700; color: #1f77b4; margin-bottom: 0.5rem; }
    .sub-header { font-size: 1rem; color: #666; margin-bottom: 2rem; }
    
    /* Chat Bubbles */
    .chat-message { padding: 1rem; border-radius: 0.75rem; margin-bottom: 0.5rem; }
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; margin-left: 20%;
    }
    .assistant-message { background: #f0f2f6; color: #1f2937; margin-right: 20%; }
    .escalation-message {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white; padding: 1rem; border-radius: 0.75rem; margin-bottom: 0.5rem;
    }
    .clarification-message {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white; padding: 1rem; border-radius: 0.75rem; margin-bottom: 0.5rem;
    }
    .thinking-message {
        background: #f9fafb;
        color: #6b7280;
        padding: 1rem;
        border-radius: 0.75rem;
        margin-bottom: 0.5rem;
        margin-right: 20%;
        font-style: italic;
    }

    /* Stats Footer under messages */
    .msg-stats {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        margin-top: 0.5rem;
        margin-bottom: 1.5rem;
        font-size: 0.8rem;
        color: #4b5563;
        display: flex;
        gap: 1.5rem;
        flex-wrap: wrap;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    .stat-item { display: flex; flex-direction: column; }
    .stat-label { font-size: 0.75rem; text-transform: uppercase; color: #9ca3af; font-weight: 700; letter-spacing: 0.05em; }
    .stat-value { font-weight: 600; color: #1f2937; font-size: 0.95rem; }
    .stat-value.escalate { color: #dc2626; }
    .stat-value.clarify { color: #2563eb; }
    .stat-value.answer { color: #059669; }

    /* Citations */
    .citation-badge {
        display: inline-block; padding: 0.2rem 0.5rem; margin: 0.1rem;
        border-radius: 0.4rem; font-size: 0.7rem; font-weight: 600;
    }
    .badge-faq { background: #dbeafe; color: #1e40af; }
    .badge-policy { background: #dcfce7; color: #166534; }
    .badge-ticket { background: #fef3c7; color: #92400e; }
    .badge-outdated { background: #fee2e2; color: #991b1b; text-decoration: line-through; }
</style>
""",
    unsafe_allow_html=True,
)

# ---------- Helper: Format Citations ----------
def format_citations(text: str) -> str:
    """Wrap citation brackets in bold green HTML for visibility."""
    pattern = r"\[(?:FAQ|POLICY|TICKET)-\d+(?:,\s*(?:FAQ|POLICY|TICKET)-\d+)*\]"
    return re.sub(
        pattern, 
        lambda m: f'<span style="color: #16a34a; font-weight: bold;">{m.group(0)}</span>', 
        text
    )

# ---------- Session state ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------- Sidebar ----------
with st.sidebar:
    # st.markdown("## 🎓 LearnForge Assistant")
    # st.markdown("*Agentic RAG powered by LangGraph*")
    # st.divider()

    if st.button("🗑️ New Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.markdown("### 💡 Try these")
    sample_queries = [
        "What's the refund policy?",
        "Can I download courses on my laptop?",
        "Cancel my LearnForge",
        "I was charged twice for a course",
        "How do I reset my password?",
    ]
    for q in sample_queries:
        if st.button(q, use_container_width=True, key=f"sample_{q}"):
            st.session_state.pending_question = q
            st.rerun()

# ---------- Main area ----------
st.markdown(
    '<div class="main-header">LearnForge Support Assistant</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-header">Ask me anything about courses, billing, refunds, or your account.</div>',
    unsafe_allow_html=True,
)

# ---------- Display chat history ----------
for msg in st.session_state.messages:
    role = msg["role"]
    
    if role == "user":
        st.markdown(
            f'<div class="chat-message user-message"><b>You:</b> {msg["content"]}</div>',
            unsafe_allow_html=True,
        )
        
    elif role == "thinking":
        st.markdown(
            f'<div class="thinking-message">🤔 Thinking...</div>',
            unsafe_allow_html=True,
        )
        
    elif role in ("assistant", "escalation", "clarification"):
        css_class = f"{role}-message"
        label = "Assistant" if role == "assistant" else ("🚨 Escalated" if role == "escalation" else " Clarification")
        
        formatted_content = format_citations(msg["content"])
        
        st.markdown(
            f'<div class="chat-message {css_class}"><b>{label}:</b> {formatted_content}</div>',
            unsafe_allow_html=True,
        )
        
        if "stats" in msg:
            s = msg["stats"]
            action_class = s['action'] 
            stats_html = f"""
            <div class="msg-stats">
                <div class="stat-item"><span class="stat-label">Confidence</span><span class="stat-value">{s['confidence']:.0%}</span></div>
                <div class="stat-item"><span class="stat-label">Intent</span><span class="stat-value">{s['intent']}</span></div>
                <div class="stat-item"><span class="stat-label">Action</span><span class="stat-value {action_class}">{s['action']}</span></div>
                <div class="stat-item"><span class="stat-label">Docs</span><span class="stat-value">{s['retrieved_docs']}</span></div>
                <div class="stat-item"><span class="stat-label">Iterations</span><span class="stat-value">{s['iterations']}</span></div>
            </div>
            """
            st.markdown(stats_html, unsafe_allow_html=True)

    elif role == "citations":
        badges_html = " ".join(
            [
                f'<span class="citation-badge badge-{c["source_type"]}'
                f'{" badge-outdated" if c.get("is_outdated") else ""}">'
                f'{c["source_id"]}</span>'
                for c in msg["citations"]
            ]
        )
        st.markdown(
            f'<div style="margin: 0 0 1.5rem 0; font-size: 0.85rem;"><b>📚 Sources:</b> {badges_html}</div>',
            unsafe_allow_html=True,
        )

# ---------- Chat input ----------
pending = st.session_state.pop("pending_question", None)
question = st.chat_input("Ask a question about LearnForge...") or pending

if question:
    # 1. Add user message
    st.session_state.messages.append({"role": "user", "content": question})
    
    # 2. Add thinking message
    st.session_state.messages.append({"role": "thinking", "content": "Thinking..."})
    
    # 3. Rerun to show user question + thinking indicator immediately
    st.rerun()

# ---------- Process answer (separate from input handling) ----------
# Check if we have a thinking message at the end
if st.session_state.messages and st.session_state.messages[-1]["role"] == "thinking":
    # Get the last user question
    user_messages = [m for m in st.session_state.messages if m["role"] == "user"]
    if user_messages:
        last_question = user_messages[-1]["content"]
        
        # Run the agent
        conversation_history = [m for m in st.session_state.messages if m["role"] in ("user", "assistant")]
        final_result = run_agent(question=last_question, messages=conversation_history)
        
        # Remove thinking message
        st.session_state.messages.pop()
        
        # Extract data
        action = final_result.get("action", "answer")
        answer = final_result.get("answer", "I couldn't generate a response.")
        
        stats = {
            "confidence": final_result.get("confidence", 0),
            "intent": final_result.get("intent", "—"),
            "action": action,
            "retrieved_docs": len(final_result.get("documents", [])),
            "iterations": final_result.get("iteration", 0),
        }
        
        # Add response with stats
        if action == "escalate":
            escalation_msg = (
                f"I'm not confident I can fully resolve this. A human agent will review your case. "
                f"Here's what I found so far: {answer}"
            )
            st.session_state.messages.append({"role": "escalation", "content": escalation_msg, "stats": stats})
        elif action == "clarify":
            st.session_state.messages.append({"role": "clarification", "content": answer, "stats": stats})
        else:
            st.session_state.messages.append({"role": "assistant", "content": answer, "stats": stats})
        
        # Add citations if any
        if final_result.get("citations"):
            st.session_state.messages.append(
                {"role": "citations", "citations": final_result["citations"]}
            )
        
        st.rerun()