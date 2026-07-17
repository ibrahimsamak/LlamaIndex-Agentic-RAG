"""Lesson 3 — Agent Reasoning Loop (OOP port of 3.ipynb).

Note: llama-index 0.14 replaced the old ``AgentRunner`` step API
(``create_task`` / ``run_step`` / ``finalize_response``) with an async
workflow. The equivalent "debuggability and control" is now streaming the
workflow's intermediate events — shown here via ``run_with_events``.
"""

from agentic_rag import RAGConfig, ReasoningAgent, load_environment


def main():
    load_environment()
    config = RAGConfig()
    config.apply_global_settings()
    llm = config.build_llm()

    agent = ReasoningAgent.for_document(llm, "metagpt.pdf", name="metagpt")

    # High-level: one-shot query, then stateful chat follow-ups.
    print(agent.query(
        "Tell me about the agent roles in MetaGPT, "
        "and then how they communicate with each other."
    ))

    agent.chat("Tell me about the evaluation datasets used.")
    print(agent.chat("Tell me the results over one of the above datasets."))

    # Low-level: stream intermediate tool-call events as the agent reasons.
    print(agent.run_with_events(
        "What about how agents share information?",
        on_event=lambda ev: print("  event:", type(ev).__name__),
    ))


if __name__ == "__main__":
    main()
