"""Agentic RAG over documents with LlamaIndex.

An OOP refactor of the DeepLearning.AI course "Building Agentic RAG with
LlamaIndex" (Lessons 1-4). Each lesson maps to one class over a shared
indexing / configuration / document-tools core:

* :class:`RouterEngine`        -- Lesson 1: route between summary and vector search
* :class:`ToolCaller`          -- Lesson 2: single ``predict_and_call`` tool selection
* :class:`ReasoningAgent`      -- Lesson 3: function-calling agent reasoning loop
* :class:`MultiDocumentAgent`  -- Lesson 4: agent over many documents w/ tool retrieval
"""

from .agent_runtime import WorkflowAgent
from .config import RAGConfig, load_environment
from .document_tools import DocumentTools
from .indexing import DocumentIndexer
from .multi_document_agent import MultiDocumentAgent
from .reasoning_agent import ReasoningAgent
from .router_engine import RouterEngine
from .tool_caller import ToolCaller

__all__ = [
    "RAGConfig",
    "load_environment",
    "DocumentIndexer",
    "DocumentTools",
    "WorkflowAgent",
    "RouterEngine",
    "ToolCaller",
    "ReasoningAgent",
    "MultiDocumentAgent",
]
