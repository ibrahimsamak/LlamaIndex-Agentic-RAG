"""Per-document tools: auto-retrieval vector search + summarization.

This is the OOP form of the course's ``get_doc_tools`` helper (Lesson 2),
reused as the tool source for the agents in Lessons 3 and 4. Each document
yields two tools:

* ``vector_tool_<name>`` — a function tool whose docstring instructs the LLM
  how to optionally filter by page number. The docstring is deliberately
  preserved verbatim: LlamaIndex feeds it to the model as the tool spec, so
  wording changes alter agent behavior.
* ``summary_tool_<name>`` — a tree-summarize query engine tool for
  whole-document questions.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from .indexing import DocumentIndexer


class DocumentTools:
    """Build and cache the (vector, summary) tool pair for one document."""

    def __init__(self, file_path: str, name: Optional[str] = None, chunk_size: int = 1024):
        self.file_path = str(file_path)
        self.name = name or Path(self.file_path).stem
        self.indexer = DocumentIndexer(self.file_path, chunk_size=chunk_size)
        self._tools = None

    def _build_vector_tool(self):
        from llama_index.core.tools import FunctionTool
        from llama_index.core.vector_stores import FilterCondition, MetadataFilters

        vector_index = self.indexer.vector_index

        def vector_query(query: str, page_numbers: Optional[List[str]] = None) -> str:
            """Use to answer questions over a given paper.

            Useful if you have specific questions over the paper.
            Always leave page_numbers as None UNLESS there is a specific page you want to search for.

            Args:
                query (str): the string query to be embedded.
                page_numbers (Optional[List[str]]): Filter by set of pages. Leave as NONE
                    if we want to perform a vector search
                    over all pages. Otherwise, filter by the set of specified pages.

            """
            page_numbers = page_numbers or []
            metadata_dicts = [{"key": "page_label", "value": p} for p in page_numbers]

            query_engine = vector_index.as_query_engine(
                similarity_top_k=2,
                filters=MetadataFilters.from_dicts(
                    metadata_dicts,
                    condition=FilterCondition.OR,
                ),
            )
            return query_engine.query(query)

        return FunctionTool.from_defaults(name=f"vector_tool_{self.name}", fn=vector_query)

    def _build_summary_tool(self):
        from llama_index.core.tools import QueryEngineTool

        summary_query_engine = self.indexer.summary_index.as_query_engine(
            response_mode="tree_summarize",
            use_async=True,
        )
        return QueryEngineTool.from_defaults(
            name=f"summary_tool_{self.name}",
            query_engine=summary_query_engine,
            description=f"Useful for summarization questions related to {self.name}",
        )

    def as_tools(self) -> list:
        """Return ``[vector_tool, summary_tool]``, building them once and caching."""
        if self._tools is None:
            self._tools = [self._build_vector_tool(), self._build_summary_tool()]
        return self._tools
