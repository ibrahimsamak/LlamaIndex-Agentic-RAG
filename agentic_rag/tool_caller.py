"""Lesson 2: Tool Calling.

The simplest agentic pattern: hand the LLM a set of tools and let a single
``predict_and_call`` decide which to invoke and with what arguments. No
reasoning loop — one selection, one call. Useful both for toy function tools
(``add``, ``mystery``) and for the document tools from :mod:`document_tools`.
"""

from __future__ import annotations

from typing import Optional, Sequence

from .document_tools import DocumentTools


class ToolCaller:
    """Wrap an LLM plus a tool set behind a single ``predict_and_call``."""

    def __init__(self, llm, tools: Optional[Sequence] = None):
        self.llm = llm
        self.tools = list(tools or [])

    def add_tool(self, tool) -> "ToolCaller":
        self.tools.append(tool)
        return self

    @classmethod
    def for_document(cls, llm, file_path: str, name: Optional[str] = None) -> "ToolCaller":
        """Build a caller pre-loaded with a document's vector and summary tools."""
        return cls(llm, DocumentTools(file_path, name=name).as_tools())

    def call(self, prompt: str, verbose: bool = True):
        """Let the LLM pick and invoke a tool for ``prompt``."""
        return self.llm.predict_and_call(self.tools, prompt, verbose=verbose)
