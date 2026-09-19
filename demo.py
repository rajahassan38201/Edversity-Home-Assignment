"""
CLI demo for the LearnForge support assistant.
Run: python demo.py
"""
from dotenv import load_dotenv

from src.graph import run_agent

load_dotenv()

BANNER = """
╔══════════════════════════════════════════════════════╗
║   🎓  LearnForge Support Assistant  (CLI demo)      ║
║   Type 'quit' to exit, 'clear' to reset history.    ║
╚══════════════════════════════════════════════════════╝
"""


def main():
    print(BANNER)
    messages = []

    while True:
        try:
            question = input("\n🧑 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() == "quit":
            print("Goodbye!")
            break
        if question.lower() == "clear":
            messages = []
            print("🗑️  Conversation cleared.")
            continue

        result = run_agent(question=question, messages=messages)
        action = result.get("action", "answer")
        answer = result.get("answer", "")
        citations = result.get("citations", [])
        confidence = result.get("confidence", 0)
        intent = result.get("intent", "")

        print()
        if action == "escalate":
            print(f"🚨 ESCALATED (confidence {confidence:.0%}, intent={intent})")
            print(f"🤖 Assistant: {answer}")
        elif action == "clarify":
            print(f"❓ CLARIFICATION (intent={intent})")
            print(f"🤖 Assistant: {answer}")
        else:
            print(f"🤖 Assistant (confidence {confidence:.0%}, intent={intent}):")
            print(answer)

        if citations:
            src_str = "  ".join(
                [
                    f"[{c['source_id']}]" + (" ⚠️ outdated" if c.get("is_outdated") else "")
                    for c in citations
                ]
            )
            print(f"\n📚 Sources: {src_str}")

        # Track conversation history
        messages.append({"role": "user", "content": question})
        messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()