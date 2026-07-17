"""Lesson 1 — Router Query Engine (OOP port of 1.ipynb)."""

from agentic_rag import RAGConfig, RouterEngine, load_environment


def main():
    load_environment()
    RAGConfig().apply_global_settings()

    engine = RouterEngine("metagpt.pdf")

    print(engine.query("What is the summary of the document?"))
    print(engine.query("How do agents share information with other agents?"))
    print(engine.query("Tell me about the ablation study results?"))


if __name__ == "__main__":
    main()
