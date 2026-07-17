"""Sync wrapper around LlamaIndex's async ``FunctionAgent`` workflow.

llama-index 0.14 replaced the old ``FunctionCallingAgentWorker`` / ``AgentRunner``
(synchronous, with ``create_task`` / ``run_step`` control) with an async
workflow: ``FunctionAgent.run()`` returns an awaitable handler that must be
driven inside a running event loop.

``WorkflowAgent`` hides that behind the synchronous ``query`` / ``chat`` methods
the rest of this project (and the course notebooks) expect. ``chat`` threads a
persistent ``Context`` so conversation memory survives across turns; ``query``
is a stateless one-shot. ``run_with_events`` exposes the intermediate workflow
events for step-level debugging — the modern analog of the old ``run_step``.

``load_environment`` applies ``nest_asyncio``, which makes ``asyncio.run``
reentrant so these wrappers work inside notebooks and already-async callers.
"""

from __future__ import annotations

import asyncio
from typing import Callable, Optional


def run_sync(coro):
    """Drive an async coroutine to completion from synchronous code."""
    return asyncio.run(coro)


class WorkflowAgent:
    """Synchronous facade over a configured ``FunctionAgent``."""

    def __init__(self, agent):
        self._agent = agent
        self._ctx = None

    def _context(self):
        from llama_index.core.workflow import Context

        if self._ctx is None:
            self._ctx = Context(self._agent)
        return self._ctx

    def query(self, question: str):
        """One-shot, stateless run. Returns an ``AgentOutput`` (``str()`` -> text)."""
        return run_sync(self._agent.run(question))

    def chat(self, message: str):
        """Run with persistent memory across calls on this instance."""
        return run_sync(self._agent.run(message, ctx=self._context()))

    def reset(self) -> None:
        """Forget conversation memory (next ``chat`` starts fresh)."""
        self._ctx = None

    def run_with_events(
        self,
        message: str,
        on_event: Optional[Callable] = None,
    ):
        """Run while streaming intermediate workflow events (tool calls, etc.).

        ``on_event`` is called with each event as it streams (defaults to
        ``print``). Returns the final ``AgentOutput``.
        """
        on_event = on_event or print

        async def _drive():
            handler = self._agent.run(message)
            async for event in handler.stream_events():
                on_event(event)
            return await handler

        return run_sync(_drive())
