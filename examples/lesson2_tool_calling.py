"""Lesson 2 — Tool Calling (OOP port of 2.ipynb)."""

from llama_index.core.tools import FunctionTool

from agentic_rag import RAGConfig, ToolCaller, load_environment


def add(x: int, y: int) -> int:
    """Adds two integers together."""
    return x + y


def mystery(x: int, y: int) -> int:
    """Mystery function that operates on top of two numbers."""
    return (x + y) * (x + y)


def main():
    load_environment()
    config = RAGConfig()
    config.apply_global_settings()
    llm = config.build_llm()

    # 1. Simple function tools
    simple = ToolCaller(
        llm,
        [FunctionTool.from_defaults(fn=add), FunctionTool.from_defaults(fn=mystery)],
    )
    print(simple.call("Tell me the output of the mystery function on 2 and 9"))

    # 2. Auto-retrieval + summary tools over the paper
    docs = ToolCaller.for_document(llm, "metagpt.pdf")
    response = docs.call(
        "What are the MetaGPT comparisons with ChatDev described on page 8?"
    )
    print(response)
    for n in response.source_nodes:
        print(n.metadata)

    print(docs.call("What is a summary of the paper?"))


if __name__ == "__main__":
    main()
