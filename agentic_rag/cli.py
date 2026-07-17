"""Command-line entry point.

Usage::

    python -m agentic_rag.cli router  --file metagpt.pdf --query "..."
    python -m agentic_rag.cli toolcall --file metagpt.pdf --query "..."
    python -m agentic_rag.cli agent    --file metagpt.pdf --query "..." [--chat]
    python -m agentic_rag.cli multi    --files a.pdf b.pdf --query "..." [--chat]

With ``--chat`` (agent / multi commands) the process drops into an interactive
REPL against the agent instead of running a single query.
"""

from __future__ import annotations

import argparse
import sys

from .config import RAGConfig, load_environment
from .multi_document_agent import MultiDocumentAgent
from .reasoning_agent import ReasoningAgent
from .router_engine import RouterEngine
from .tool_caller import ToolCaller


def _repl(respond) -> None:
    """Read questions from stdin and print responses until EOF or 'exit'."""
    print("Interactive chat — type 'exit' or Ctrl-D to quit.")
    while True:
        try:
            message = input("\n> ").strip()
        except EOFError:
            break
        if message.lower() in {"exit", "quit"}:
            break
        if message:
            print(str(respond(message)))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="agentic_rag", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p, multi=False):
        if multi:
            p.add_argument("--files", nargs="+", required=True, help="Document paths")
        else:
            p.add_argument("--file", required=True, help="Document path")
        p.add_argument("--query", help="Question to ask")

    add_common(sub.add_parser("router", help="Lesson 1: router query engine"))
    add_common(sub.add_parser("toolcall", help="Lesson 2: single tool call"))

    agent_p = sub.add_parser("agent", help="Lesson 3: reasoning agent")
    add_common(agent_p)
    agent_p.add_argument("--chat", action="store_true", help="Interactive chat loop")

    multi_p = sub.add_parser("multi", help="Lesson 4: multi-document agent")
    add_common(multi_p, multi=True)
    multi_p.add_argument("--chat", action="store_true", help="Interactive chat loop")

    args = parser.parse_args(argv)

    load_environment()
    config = RAGConfig()
    config.apply_global_settings()
    llm = config.build_llm()

    if args.command == "router":
        engine = RouterEngine(args.file, chunk_size=config.chunk_size)
        print(str(engine.query(args.query)))

    elif args.command == "toolcall":
        caller = ToolCaller.for_document(llm, args.file)
        print(str(caller.call(args.query)))

    elif args.command == "agent":
        agent = ReasoningAgent.for_document(llm, args.file)
        if args.chat:
            _repl(agent.chat)
        else:
            print(str(agent.query(args.query)))

    elif args.command == "multi":
        agent = MultiDocumentAgent(args.files, llm)
        if args.chat:
            _repl(agent.chat)
        else:
            print(str(agent.query(args.query)))

    return 0


if __name__ == "__main__":
    sys.exit(main())
