"""Lesson 1: Router Query Engine.

Builds a summary index and a vector index over the same document, wraps each
in a query-engine tool, and lets an ``LLMSingleSelector`` pick which one to use
per query — summary tool for "what is this about" questions, vector tool for
specific-fact lookups.
"""

from __future__ import annotations

from .indexing import DocumentIndexer


class RouterEngine:
    """A single-document router that dispatches queries to summary or vector search."""

    def __init__(self, file_path: str, chunk_size: int = 1024, verbose: bool = True):
        self.indexer = DocumentIndexer(file_path, chunk_size=chunk_size)
        self.verbose = verbose
        self._engine = None

    def _build(self):
        from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
        from llama_index.core.selectors import LLMSingleSelector
        from llama_index.core.tools import QueryEngineTool

        summary_query_engine = self.indexer.summary_index.as_query_engine(
            response_mode="tree_summarize",
            use_async=True,
        )
        vector_query_engine = self.indexer.vector_index.as_query_engine()

        summary_tool = QueryEngineTool.from_defaults(
            query_engine=summary_query_engine,
            description="Useful for summarization questions related to the document.",
        )
        vector_tool = QueryEngineTool.from_defaults(
            query_engine=vector_query_engine,
            description="Useful for retrieving specific context from the document.",
        )

        return RouterQueryEngine(
            selector=LLMSingleSelector.from_defaults(),
            query_engine_tools=[summary_tool, vector_tool],
            verbose=self.verbose,
        )

    @property
    def engine(self):
        if self._engine is None:
            self._engine = self._build()
        return self._engine

    def query(self, question: str):
        return self.engine.query(question)
