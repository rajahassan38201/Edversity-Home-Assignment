"""
LangGraph nodes for the LearnForge support assistant.
Uses Mistral AI for both LLM and Embeddings.
"""
import os
import re
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

def get_llm():
    return ChatMistralAI(
        model="open-mistral-nemo",
        temperature=0,
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
    )

def get_vectorstore():
    embeddings = MistralAIEmbeddings(
        model="mistral-embed",
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
    )
    return Chroma(
        collection_name="learnforge_knowledge",
        embedding_function=embeddings,
        persist_directory="./chroma_db",
    )

def router_node(state: Dict[str, Any]) -> Dict[str, Any]:
    question = state["question"]
    messages = state.get("messages", [])
    
    prompt = ChatPromptTemplate.from_template(
        """You are a query router for the LearnForge customer support assistant.
        
Conversation history: {history}
Current user question: {question}

Classify the question into EXACTLY ONE category:
- direct: ONLY for greetings (hi, hello), thanks, or questions about YOU (e.g., "who are you?", "what is your name?").
- retrieve: ANY question about LearnForge products, accounts, billing, courses, passwords, refunds, or policies. Even if you think you know the answer generally, you MUST retrieve the specific LearnForge policy.
- ambiguous: Unclear what the user wants (e.g., "Cancel my LearnForge" - could mean subscription, enrollment, or account deletion).
- billing_dispute: User disputes a charge, claims unauthorized transaction, or demands a refund outside standard policy.

Respond with ONLY the category name (direct, retrieve, ambiguous, or billing_dispute)."""
    )
    
    history = "\n".join([f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}" for m in messages[-6:]]) if messages else "No prior conversation."
    
    chain = prompt | get_llm()
    result = chain.invoke({"question": question, "history": history})
    category = result.content.strip().lower()
    
    if category not in ("direct", "retrieve", "ambiguous", "billing_dispute"):
        category = "retrieve"
        
    return {"intent": category}

def retriever_node(state: Dict[str, Any]) -> Dict[str, Any]:
    question = state["question"]
    docs = get_vectorstore().similarity_search(question, k=5)
    retrieved = [{
        "content": doc.page_content,
        "source_id": doc.metadata.get("source_id"),
        "source_type": doc.metadata.get("source_type"),
        "topic": doc.metadata.get("topic"),
        "is_outdated": doc.metadata.get("is_outdated", False),
        "status": doc.metadata.get("status"),
        "last_reviewed": doc.metadata.get("last_reviewed"),
    } for doc in docs]
    return {"documents": retrieved}

def generator_node(state: Dict[str, Any]) -> Dict[str, Any]:
    question = state["question"]
    docs = state.get("documents", [])
    messages = state.get("messages", [])
    intent = state.get("intent", "retrieve")

    context_parts = [f"[{doc['source_id']}]{ ' [OUTDATED]' if doc['is_outdated'] else ''} ({doc['source_type'].upper()}):\n{doc['content']}" for doc in docs]
    context = "\n\n---\n\n".join(context_parts) if context_parts else "No documents retrieved."
    history = "\n".join([f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}" for m in messages[-6:]]) if messages else "No prior conversation."

    if intent == "ambiguous":
        system_msg = "You are a customer support agent. The user's request is ambiguous. Ask ONE clarifying question to understand what they need. Be concise and friendly. Do NOT answer yet or cite sources."
    elif intent == "direct":
        system_msg = """You are the LearnForge Support Assistant, an AI-powered help desk. 
For greetings or questions about yourself: Briefly introduce yourself and invite them to ask about courses, billing, refunds, or accounts.
For other general questions: Provide a helpful, friendly, brief response. No citations needed.
Conversation history: {history}
User question: {question}
Response:"""
    else:
        system_msg = """You are a customer support agent for LearnForge. Answer using ONLY the provided context.
CRITICAL RULES:
1. Cite every factual claim using [SOURCE-ID] (e.g., [FAQ-02], [POLICY-03]).
2. If sources conflict, mention both sides and cite each.
3. If a source is marked [OUTDATED], do NOT use it as the primary answer — note that it may be stale.
4. If the context doesn't contain the answer, say so clearly.
5. Do not make up information.

Conversation history: {history}
Context: {context}
User question: {question}
Answer:"""

    prompt = ChatPromptTemplate.from_template(system_msg)
    chain = prompt | get_llm()
    result = chain.invoke({"history": history, "context": context, "question": question})
    answer = result.content.strip()

    citation_ids = re.findall(r"\[(FAQ|POLICY|TICKET)-\d+\]", answer)
    citations = []
    seen = set()
    for cid in citation_ids:
        if cid not in seen:
            seen.add(cid)
            for doc in docs:
                if doc["source_id"] == cid:
                    citations.append({"source_id": cid, "source_type": doc["source_type"], "is_outdated": doc["is_outdated"]})
                    break
    return {"answer": answer, "citations": citations}

def grader_node(state: Dict[str, Any]) -> Dict[str, Any]:
    question = state["question"]
    answer = state.get("answer", "")
    docs = state.get("documents", [])
    intent = state.get("intent", "retrieve")
    if intent in ("ambiguous", "direct"):
        return {"confidence": 0.9, "grade": "pass"}
    
    context = "\n\n".join([d["content"] for d in docs[:3]])
    prompt = ChatPromptTemplate.from_template(
        """You are a quality grader. Evaluate if the answer is grounded in the context.
Question: {question}
Context: {context}
Answer: {answer}
Score 0.0 to 1.0: 1.0=Fully grounded, 0.7-0.9=Mostly grounded, 0.4-0.6=Partially grounded, 0.0-0.3=Hallucinated.
Respond with ONLY the numeric score (e.g., 0.85)."""
    )
    chain = prompt | get_llm()
    result = chain.invoke({"question": question, "context": context, "answer": answer})
    try:
        score = max(0.0, min(1.0, float(result.content.strip())))
    except ValueError:
        score = 0.5
    return {"confidence": score, "grade": "pass" if score >= 0.7 else "fail"}

def confidence_router_node(state: Dict[str, Any]) -> Dict[str, Any]:
    intent = state.get("intent", "retrieve")
    confidence = state.get("confidence", 0.0)
    grade = state.get("grade", "pass")
    iteration = state.get("iteration", 0)

    if intent == "ambiguous": return {"action": "clarify"}
    if intent == "direct": return {"action": "answer"}
    if intent == "billing_dispute": return {"action": "escalate"}
    if grade == "fail" and iteration >= 2: return {"action": "escalate"}
    if confidence < 0.5: return {"action": "escalate"}
    if confidence < 0.7 and grade == "fail": return {"action": "retry"}
    return {"action": "answer"}