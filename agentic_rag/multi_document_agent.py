"""Lesson 4: Multi-Document Agent.

Scales the Lesson 3 pattern from one document to many. Each document
contributes a (vector, summary) tool pair, so N papers means 2N tools —
too many to stuff into a prompt. Past a small number of documents this
switches to *tool retrieval*: the tools are themselves indexed in an
``ObjectIndex`` and only the top-k most relevant are retrieved per query.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

from .agent_runtime import WorkflowAgent
from .document_tools import DocumentTools

DEFAULT_SYSTEM_PROMPT = """ \
You are an agent designed to answer queries over a set of given papers.
Please always use the tools provided to answer a question. Do not rely on prior knowledge.\

"""


class MultiDocumentAgent(WorkflowAgent):
    """A function-calling agent over many documents, with optional tool retrieval.

    When ``use_tool_retrieval`` is left as ``None`` it is enabled automatically
    for more than three documents (matching the course's cutoff between the
    "3 papers" and "11 papers" setups). With retrieval on, tools are selected
    per query via an object index instead of all being passed to the LLM.
    """

    def __init__(
        self,
        file_paths: Sequence[str],
        llm,
        similarity_top_k: int = 3,
        use_tool_retrieval: Optional[bool] = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ):
        from llama_index.core.agent.workflow import FunctionAgent

        self.file_paths = [str(p) for p in file_paths]
        self.tools = []
        for path in self.file_paths:
            print(f"Getting tools for paper: {path}")
            self.tools.extend(DocumentTools(path, name=Path(path).stem).as_tools())

        if use_tool_retrieval is None:
            use_tool_retrieval = len(self.file_paths) > 3

        if use_tool_retrieval:
            agent = FunctionAgent(
                tool_retriever=self._build_tool_retriever(similarity_top_k),
                llm=llm,
                system_prompt=system_prompt,
            )
        else:
            agent = FunctionAgent(
                tools=self.tools,
                llm=llm,
                system_prompt=system_prompt,
            )

        super().__init__(agent)

    def _build_tool_retriever(self, similarity_top_k: int):
        from llama_index.core import VectorStoreIndex
        from llama_index.core.objects import ObjectIndex

        obj_index = ObjectIndex.from_objects(self.tools, index_cls=VectorStoreIndex)
        return obj_index.as_retriever(similarity_top_k=similarity_top_k)
