"""Lesson 3: Agent Reasoning Loop.

Wraps document tools in a ``FunctionAgent`` (llama-index 0.14's function-calling
agent). Unlike Lesson 2's single ``predict_and_call``, the agent loops — calling
tools, reading results, and deciding follow-up actions until it can answer.

Interface (see :class:`~agentic_rag.agent_runtime.WorkflowAgent`):

* :meth:`query`           -- one-shot, stateless.
* :meth:`chat`            -- keeps conversation memory across turns.
* :meth:`run_with_events` -- stream intermediate tool-call events for
  step-level debugging (the modern replacement for the old ``run_step`` control
  shown in the notebook's "Debuggability and Control" section).
"""

from __future__ import annotations

from typing import Optional, Sequence

from .agent_runtime import WorkflowAgent
from .document_tools import DocumentTools


class ReasoningAgent(WorkflowAgent):
    """A function-calling agent over one or more tools."""

    def __init__(
        self,
        tools: Sequence,
        llm,
        system_prompt: Optional[str] = None,
    ):
        from llama_index.core.agent.workflow import FunctionAgent

        agent = FunctionAgent(tools=list(tools), llm=llm, system_prompt=system_prompt)
        super().__init__(agent)

    @classmethod
    def for_document(
        cls,
        llm,
        file_path: str,
        name: Optional[str] = None,
        **kwargs,
    ) -> "ReasoningAgent":
        """Build an agent over one document's vector and summary tools."""
        return cls(DocumentTools(file_path, name=name).as_tools(), llm, **kwargs)
