"""
LangGraph orchestration for the LearnForge support assistant.

Flow:
  Router → (Retriever → Generator) OR Generator
       → Grader → Confidence Router
       → (Answer | Clarify | Escalate | Retry)
"""
from typing import Any, Dict, List, Literal, TypedDict

from langgraph.graph import END, StateGraph

from src.nodes import (
    confidence_router_node,
    generator_node,
    grader_node,
    retriever_node,
    router_node,
)


class AgentState(TypedDict):
    question: str
    messages: List[Dict[str, str]]
    intent: str
    documents: List[Dict[str, Any]]
    answer: str
    citations: List[Dict[str, Any]]
    confidence: float
    grade: str
    action: str
    iteration: int


def should_retry(state: AgentState) -> Literal["retry", "finish"]:
    """Decide whether to retry retrieval or finish."""
    action = state.get("action", "answer")
    iteration = state.get("iteration", 0)

    if action in ("clarify", "escalate", "answer"):
        return "finish"
    if action == "retry" and iteration < 2:
        return "retry"
    return "finish"


def increment_iteration(state: AgentState) -> Dict[str, Any]:
    """Increment iteration counter before retry."""
    return {"iteration": state.get("iteration", 0) + 1}


def build_graph():
    """Build and compile the LangGraph."""
    workflow = StateGraph(AgentState)

    workflow.add_node("router", router_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("generator", generator_node)
    workflow.add_node("grader", grader_node)
    workflow.add_node("confidence_router", confidence_router_node)
    workflow.add_node("increment_iteration", increment_iteration)

    workflow.set_entry_point("router")

    def route_after_router(state: AgentState) -> Literal["retriever", "generator"]:
        intent = state.get("intent", "retrieve")
        if intent in ("retrieve", "billing_dispute"):
            return "retriever"
        return "generator"

    workflow.add_conditional_edges(
        "router",
        route_after_router,
        {"retriever": "retriever", "generator": "generator"},
    )

    workflow.add_edge("retriever", "generator")
    workflow.add_edge("generator", "grader")
    workflow.add_edge("grader", "confidence_router")

    workflow.add_conditional_edges(
        "confidence_router",
        should_retry,
        {
            "retry": "increment_iteration",
            "finish": END,
        },
    )

    workflow.add_edge("increment_iteration", "retriever")

    return workflow.compile()


# Singleton graph
graph = build_graph()


def run_agent(question: str, messages: List[Dict[str, str]] = None) -> Dict[str, Any]:
    """Run the agent and return the final state."""
    if messages is None:
        messages = []

    initial_state: AgentState = {
        "question": question,
        "messages": messages,
        "intent": "",
        "documents": [],
        "answer": "",
        "citations": [],
        "confidence": 0.0,
        "grade": "",
        "action": "",
        "iteration": 0,
    }

    return graph.invoke(initial_state)


def stream_agent(question: str, messages: List[Dict[str, str]] = None):
    """Stream node-level progress events. Yields (node_name, node_output) tuples."""
    if messages is None:
        messages = []

    initial_state: AgentState = {
        "question": question,
        "messages": messages,
        "intent": "",
        "documents": [],
        "answer": "",
        "citations": [],
        "confidence": 0.0,
        "grade": "",
        "action": "",
        "iteration": 0,
    }

    for event in graph.stream(initial_state, stream_mode="updates"):
        for node_name, node_output in event.items():
            yield node_name, node_output