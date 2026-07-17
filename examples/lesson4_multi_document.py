"""Lesson 4 — Multi-Document Agent (OOP port of 4.ipynb).

Requires the papers to be present locally. The course downloads them with,
e.g.::

    wget "https://openreview.net/pdf?id=VtmBAGCN7o" -O metagpt.pdf
    wget "https://openreview.net/pdf?id=6PmJoRfdaK" -O longlora.pdf
    wget "https://openreview.net/pdf?id=hSyW5go0v8" -O selfrag.pdf
"""

from agentic_rag import MultiDocumentAgent, RAGConfig, load_environment


def main():
    load_environment()
    config = RAGConfig()
    config.apply_global_settings()
    llm = config.build_llm()

    papers = ["metagpt.pdf", "longlora.pdf", "selfrag.pdf"]

    # 3 papers: all tools passed directly (retrieval auto-off at <= 3 docs).
    agent = MultiDocumentAgent(papers, llm)
    print(agent.query("Give me a summary of both Self-RAG and LongLoRA"))

    # For many papers, force tool retrieval so only the top-k tools per query
    # reach the LLM:
    #   agent = MultiDocumentAgent(many_papers, llm, use_tool_retrieval=True)


if __name__ == "__main__":
    main()
