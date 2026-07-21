# Agentic RAG — Session Explainer

A walkthrough of the four lesson pipelines in this project, plus the Python
concepts (`@classmethod`, `@property`, `super().__init__` ordering) that show up
in the code. All examples reference the real files in `agentic_rag/`.

---

## Table of contents

1. [Lesson 1 — Router Query Engine](#lesson-1--router-query-engine)
2. [Lesson 2 — Tool Calling](#lesson-2--tool-calling)
3. [Lesson 3 — Reasoning Agent](#lesson-3--reasoning-agent)
4. [Lesson 4 — Multi-Document Agent](#lesson-4--multi-document-agent)
5. [The arc across all four lessons](#the-arc-across-all-four-lessons)
6. [Python concept: `@classmethod`](#python-concept-classmethod)
7. [Python concept: `@property`](#python-concept-property)
8. [Python concept: why `super().__init__(agent)` comes last](#python-concept-why-superinitagent-comes-last)

---

## Lesson 1 — Router Query Engine

`examples/lesson1_router.py`

### Overview
The simplest form of "agentic" behavior: two different query engines over the
*same* PDF, and an LLM picks which one to use per question — a single routing
decision, no tool-calling loop.

### Call chain
```
lesson1_router.py
  ├─ load_environment()                      config.py:16
  ├─ RAGConfig().apply_global_settings()     config.py:61
  ├─ RouterEngine("metagpt.pdf")             router_engine.py:17
  └─ engine.query(question)                  router_engine.py:54
       └─ RouterEngine.engine (lazy) → _build()   router_engine.py:22
            └─ DocumentIndexer.summary_index / vector_index   indexing.py
```

### Step by step
1. **`load_environment()`** (config.py:16) — `load_dotenv()` reads `.env` into the
   process env; `nest_asyncio.apply()` patches the event loop so the summary
   engine's `use_async=True` calls run inside an already-running loop. Raises if
   `OPENAI_API_KEY` is missing.
2. **`RAGConfig().apply_global_settings()`** (config.py:61) — installs
   `Settings.llm` (`gpt-3.5-turbo`, temp 0.0) and `Settings.embed_model`
   (`text-embedding-ada-002`) as **globals**. The router/selector/query engines
   never get an explicit LLM, so they read these globals. This is the *only*
   place a provider SDK (OpenAI) is chosen.
3. **`RouterEngine("metagpt.pdf")`** (router_engine.py:17) — cheap constructor.
   Only creates a `DocumentIndexer` and sets `_engine = None`. No PDF read, no API
   call yet — everything is lazy.
4. **First `engine.query(...)`** triggers the lazy `.engine` property → `_build()`:
   - **Parse + index** via `DocumentIndexer`: `.nodes` (SimpleDirectoryReader →
     SentenceSplitter, ~1024-token nodes), `.summary_index` (SummaryIndex, no
     embeddings), `.vector_index` (VectorStoreIndex, **embeds every node**). Both
     indexes reuse the same cached nodes → PDF parsed once.
   - **Wrap each index** as a query engine + `QueryEngineTool`. The **description
     strings** (router_engine.py:35, 39) are what the selector LLM reads to route —
     they're prompts, not comments.
   - **Assemble** a `RouterQueryEngine` with an `LLMSingleSelector`.
5. **Query execution** (router_engine.py:54) — for each query:
   1. `LLMSingleSelector` sends question + tool descriptions to the LLM → returns
      one choice.
   2. The chosen engine runs: summary tool = tree-summarize over all nodes;
      vector tool = embed query, top-k retrieval, synthesize.
   3. Returns a `Response`; `verbose=True` prints the selected tool.

The three example queries are chosen to exercise routing: one summary-style, two
specific-fact lookups.

### Key notes
- Global `Settings` is the wiring mechanism — skipping step 2 breaks routing.
- Everything is lazy and cached — cost is deferred to first query, paid once.
- `metagpt.pdf` must exist locally or it fails at first query.

---

## Lesson 2 — Tool Calling

`examples/lesson2_tool_calling.py`

### Overview
The first genuinely agentic step. Instead of Lesson 1's `RouterQueryEngine`
(which picks one *query engine* via a selector), Lesson 2 hands the LLM a set of
**tools** and uses `llm.predict_and_call(...)` — native OpenAI function calling.
The LLM decides which tool to invoke **and generates the arguments**. Still a
single selection, no loop (that's Lesson 3).

Two distinctions from Lesson 1:
1. The LLM fills in **structured arguments** (e.g. `page_numbers=["8"]`).
2. It works on both toy Python functions *and* document tools.

### Call chain
```
lesson2_tool_calling.py
  ├─ load_environment()                      config.py:16
  ├─ config.apply_global_settings()          config.py:61
  ├─ llm = config.build_llm()                config.py:51   (explicit LLM handle)
  ├─ Part 1: ToolCaller(llm, [FunctionTool(add), FunctionTool(mystery)])
  │            simple.call(...) → llm.predict_and_call(...)   tool_caller.py:32
  └─ Part 2: ToolCaller.for_document(llm, "metagpt.pdf")      tool_caller.py:27
               └─ DocumentTools(...).as_tools()               document_tools.py:78
                    ├─ _build_vector_tool()                   document_tools.py:32
                    └─ _build_summary_tool()                  document_tools.py:65
               docs.call(...) → llm.predict_and_call(...)
```

### Step by step
1. **Setup** (lines 19–22) — `load_environment()`, `apply_global_settings()`
   (needed because the document query engines read global `Settings`), and
   `llm = config.build_llm()`. The **explicit** LLM handle is needed here because
   `predict_and_call` is a method *on the LLM object* (Lesson 1 never needed it —
   the router pulled the LLM from `Settings` internally).
2. **Part 1 — toy function tools** (lines 24–29): `add` and `mystery` are plain
   Python functions. Their **docstrings and type hints matter** —
   `FunctionTool.from_defaults(fn=...)` introspects the signature and docstring to
   build the JSON tool schema. `simple.call(...)` → `predict_and_call`: the LLM
   picks `mystery`, emits `{x: 2, y: 9}`, LlamaIndex invokes the real function →
   `(2+9)*(2+9) = 121`. The LLM never computes the math; it routes to the tool.
3. **Part 2 — document tools** (lines 31–40): `ToolCaller.for_document` builds a
   caller from `DocumentTools.as_tools()` — **the pivot object** Lessons 2, 3, 4
   all reuse. Two tools:
   - **`vector_tool_metagpt`** (document_tools.py:32) — a `FunctionTool` wrapping
     `vector_query(query, page_numbers=None)`. Its **docstring is load-bearing**
     (passed verbatim to the LLM as the tool spec) and instructs the model to
     leave `page_numbers=None` unless a page is named. On call it builds a vector
     engine with `similarity_top_k=2` and a **metadata filter** on `page_label`,
     so `page_numbers=["8"]` becomes a real retrieval filter.
   - **`summary_tool_metagpt`** (document_tools.py:65) — a `QueryEngineTool` over
     the summary index (`tree_summarize`, `use_async=True`).

   The queries: the "page 8" one makes the LLM emit `page_numbers=["8"]` → top-2
   search filtered to page 8; lines 37–38 print `n.metadata` to confirm the
   chunks came from page 8. The "summary" one routes to the summary tool.

### Contrast with Lesson 1
| | Lesson 1 (`RouterEngine`) | Lesson 2 (`ToolCaller`) |
|---|---|---|
| Selection | `LLMSingleSelector` over query engines | `llm.predict_and_call` (function calling) |
| LLM output | just a route (index) | tool **+ generated arguments** |
| Vector tool | plain top-k engine | `vector_query` fn with **page-filter arg** |
| LLM handle | pulled from `Settings` | explicit `llm` object required |
| Reasoning | one shot | one shot |

The new capability is **structured tool arguments** — turning "on page 8" into a
real metadata filter.

---

## Lesson 3 — Reasoning Agent

`examples/lesson3_reasoning_agent.py`

### Overview
The **agent reasoning loop**. Lesson 2 did one `predict_and_call`. Lesson 3 wraps
the same document tools in a **`FunctionAgent`** that *loops*: call a tool, read
the result, decide whether to call another, repeat, then answer. That loop is
what lets one query ("roles **and then** how they communicate") trigger multiple
tool calls.

Second theme: the **0.14 API migration**. The old `AgentRunner`
(`create_task`/`run_step`/`finalize_response`) is gone, replaced by the **async**
`FunctionAgent.run()` workflow. `WorkflowAgent` is a synchronous facade over it.

### Call chain
```
lesson3_reasoning_agent.py
  ├─ load_environment()               config.py:16
  ├─ config.apply_global_settings()   config.py:61
  ├─ llm = config.build_llm()         config.py:51
  ├─ ReasoningAgent.for_document(llm, "metagpt.pdf", name="metagpt")
  │     ├─ DocumentTools(...).as_tools()      document_tools.py:78
  │     ├─ FunctionAgent(tools, llm, ...)     reasoning_agent.py:35
  │     └─ WorkflowAgent.__init__(agent)      agent_runtime.py:32
  ├─ agent.query(...)            → run_sync(agent.run(q))                 agent_runtime.py:43
  ├─ agent.chat(...) x2          → run_sync(agent.run(msg, ctx=self._ctx)) agent_runtime.py:47
  └─ agent.run_with_events(...)  → stream_events() then await handler     agent_runtime.py:55
```

### Step by step
1. **Setup** (lines 13–16) — same as Lesson 2. Here `nest_asyncio` is *essential*:
   the class drives async coroutines via `asyncio.run` from sync methods, and
   `nest_asyncio` makes that reentrant.
2. **Build the agent** (line 18) — `ReasoningAgent.for_document` calls
   `DocumentTools(...).as_tools()` (the same pivot pair) and passes them +
   the LLM into `FunctionAgent`. `super().__init__(agent)` stores it in
   `WorkflowAgent` with `_ctx = None`. Inheritance split: `ReasoningAgent` handles
   *construction*; `WorkflowAgent` handles *execution* (sync-over-async).
3. **`agent.query(...)`** (agent_runtime.py:43) — `run_sync(self._agent.run(q))`.
   `run()` returns an awaitable handler (0.14 async API); `asyncio.run` drives it.
   The **reasoning loop**: call vector tool for "roles" → read result → call
   vector tool again for "communicate" → final answer. No `ctx` → **stateless**.
4. **`agent.chat(...)`** (agent_runtime.py:47) — passes a persistent
   `Context(self._agent)`, cached and reused across turns, so "one of the **above**
   datasets" resolves against prior memory. `reset()` drops `_ctx`. `query` never
   touches `_ctx`, so one-shot and chat memory stay independent.
5. **`agent.run_with_events(...)`** (agent_runtime.py:55) — the modern replacement
   for the old `run_step` control. The handler is both an async iterator
   (`stream_events()`) and an awaitable (final result): consume the stream
   (printing each event class name — `AgentInput`, `ToolCall`, `ToolCallResult`,
   `AgentOutput`), then `await` for the answer. Runs **without** `ctx`, so it's
   stateless like `query`.

### Contrast
| | L1 `RouterEngine` | L2 `ToolCaller` | L3 `ReasoningAgent` |
|---|---|---|---|
| Mechanism | `LLMSingleSelector` | `llm.predict_and_call` | `FunctionAgent.run()` workflow |
| Tool calls per query | 1 route | 1 call | **loops — many calls** |
| Multi-step queries | ✗ | ✗ | ✓ |
| Memory across turns | ✗ | ✗ | ✓ (`chat` + `Context`) |
| Execution model | sync | sync | **async, wrapped sync** |
| Observability | `verbose` prints | `verbose` prints | **event streaming** |

### Version-critical detail
llama-index 0.14 **deleted** `FunctionCallingAgentWorker`/`AgentRunner` and the
`create_task`/`run_step`/`finalize_response` control. `agent.run()` now needs a
running event loop and returns an awaitable. `WorkflowAgent` bridges the gap —
`run_sync` (`asyncio.run`, made reentrant by `nest_asyncio`) keeps the sync
`query`/`chat` API; `stream_events()` recovers step-level visibility. Don't
reintroduce the old API.

---

## Lesson 4 — Multi-Document Agent

`examples/lesson4_multi_document.py`

### Overview
**Lesson 3 scaled from one document to many.** The reasoning-loop agent is
unchanged; what's new is the **tool-count problem**. Each paper contributes a
`(vector, summary)` pair, so N papers = **2N tools** — fine for 3 papers (6
tools), breaks at 11 (22 tools). The fix is **tool retrieval**: index the tools
themselves in an `ObjectIndex` and retrieve only the top-k relevant tools per
query, instead of stuffing all of them into the prompt.

### Call chain
```
lesson4_multi_document.py
  ├─ load_environment()               config.py:16
  ├─ config.apply_global_settings()   config.py:61
  ├─ llm = config.build_llm()         config.py:51
  └─ MultiDocumentAgent(papers, llm)              multi_document_agent.py:34
       ├─ for each paper: DocumentTools(...).as_tools()   document_tools.py:78
       ├─ decide use_tool_retrieval  (auto: > 3 docs)     multi_document_agent.py:50
       ├─ FunctionAgent(tools=... OR tool_retriever=...)  multi_document_agent.py:53
       └─ WorkflowAgent.__init__(agent)                   agent_runtime.py:32
  └─ agent.query(...)  → run_sync(agent.run(q))           agent_runtime.py:43
```

### Step by step
1. **Setup** (lines 15–18) — identical to Lessons 2–3.
2. **Build the agent** (line 23) — `MultiDocumentAgent(papers, llm)`:
   - **Gather tools per paper** (lines 44–48): each paper → its own
     `DocumentIndexer` → `[vector_tool_<stem>, summary_tool_<stem>]`. Tools are
     **name-disambiguated by file stem** (`vector_tool_metagpt`,
     `summary_tool_longlora`, ...) so the LLM/retriever can tell papers apart. 3
     papers → 6 tools.
   - **Decide retrieval mode** (lines 50–51): `use_tool_retrieval is None` →
     auto-decide by count. **≤ 3 docs → OFF.** Mirrors the course's 3-vs-11 split.
   - **Construct the `FunctionAgent`** — two branches:
     - **OFF (taken here, lines 60–64):** `FunctionAgent(tools=self.tools, ...)` —
       all 6 tools in the prompt. Exactly the Lesson 3 setup, just 6 tools/3 papers.
     - **ON (lines 53–58):** `FunctionAgent(tool_retriever=..., ...)`.
       `_build_tool_retriever` embeds each **tool** into a `VectorStoreIndex` via
       `ObjectIndex.from_objects`, then `as_retriever(similarity_top_k=3)`. Per
       query, only the top-3 relevant tools reach the LLM — the scaling mechanism.
   - `DEFAULT_SYSTEM_PROMPT` (lines 18–22) — new in L4: "always use the tools, do
     not rely on prior knowledge" — guards against answering from memory when many
     similar papers are present.
   - `super().__init__(agent)` (line 66) wires it into `WorkflowAgent`.
3. **Run the query** (line 24) — `agent.query("summary of both Self-RAG and
   LongLoRA")`: stateless sync-over-async run. The **reasoning loop spans
   documents**: call `summary_tool_selfrag` → call `summary_tool_longlora` →
   combine. Disambiguated names let the LLM target the right paper's tool.

### Contrast: Lesson 3 vs Lesson 4
| | L3 `ReasoningAgent` | L4 `MultiDocumentAgent` |
|---|---|---|
| Documents | 1 | many |
| Tools | 2 | 2N (name-scoped per paper) |
| Tool delivery | all in prompt | all **or** top-k via `ObjectIndex` retriever |
| Retrieval switch | n/a | auto-on when > 3 docs |
| System prompt | none | "always use tools, no prior knowledge" |
| Loop / runtime | `FunctionAgent` + `WorkflowAgent` | **same** |

Lesson 4 adds exactly two ideas: **per-document tool naming** and **tool
retrieval** (`ObjectIndex`). Everything below is reused unchanged.

### Practical note
All three PDFs must exist locally or construction fails at the first
`DocumentTools` call:
```
wget "https://openreview.net/pdf?id=VtmBAGCN7o" -O metagpt.pdf
wget "https://openreview.net/pdf?id=6PmJoRfdaK" -O longlora.pdf
wget "https://openreview.net/pdf?id=hSyW5go0v8" -O selfrag.pdf
```

---

## The arc across all four lessons

**route (L1) → call one tool (L2) → loop over tools (L3) → scale the loop across
many documents with tool retrieval (L4)** — each layer composes the one below it.
`DocumentTools`, `FunctionAgent`, and the async `WorkflowAgent` facade are reused
unchanged as the lessons build up.

---

## Python concept: `@classmethod`

### What it is
A normal method receives the **instance** (`self`) as its first argument. A
`@classmethod` receives the **class** (`cls`) instead. It's called on the class,
not an instance, and doesn't need an existing object.

```python
class Foo:
    def normal(self): ...        # gets the instance
    @classmethod
    def factory(cls): ...        # gets the class (Foo)

Foo().normal()   # called on an instance
Foo.factory()    # called on the class — no instance needed
```

### Codebase example — `ReasoningAgent.for_document` (reasoning_agent.py:38)
```python
class ReasoningAgent(WorkflowAgent):
    def __init__(self, tools, llm, system_prompt=None):
        agent = FunctionAgent(tools=list(tools), llm=llm, system_prompt=system_prompt)
        super().__init__(agent)

    @classmethod
    def for_document(cls, llm, file_path, name=None, **kwargs):
        return cls(DocumentTools(file_path, name=name).as_tools(), llm, **kwargs)
```
Two ways to build the same object:
```python
tools = DocumentTools("metagpt.pdf", name="metagpt").as_tools()
agent = ReasoningAgent(tools, llm)                              # verbose
agent = ReasoningAgent.for_document(llm, "metagpt.pdf", name="metagpt")  # convenient
```
`for_document` is an **alternative constructor**: `__init__` takes low-level
pieces (a ready tool list); the classmethod takes high-level inputs (a file path),
does the assembly, then calls `cls(...)`.

### Why use it
1. **Alternative constructors** (the main reason, and this codebase's use). Like
   `dict.fromkeys()`, `datetime.now()`. Both `ToolCaller.for_document`
   (tool_caller.py:27) and `ReasoningAgent.for_document` are this.
2. **`cls` respects inheritance** — a subclass gets *its own type* back:
   ```python
   class SpecialAgent(ReasoningAgent): ...
   x = SpecialAgent.for_document(llm, "metagpt.pdf")  # builds a SpecialAgent
   ```
   Hardcoding `return ReasoningAgent(...)` would break this. `cls` makes it
   polymorphic for free.
3. **Class-level work needing no instance** — a factory can't use `self` because
   the object doesn't exist until the classmethod builds it.

### When to use which
- `@classmethod` (`cls`) — alternative constructor / class-level work.
- normal method (`self`) — works on one object's data.
- `@staticmethod` (neither) — related utility touching neither instance nor class.

```python
class ToolCaller:
    def __init__(self, llm, tools=None):        # low-level constructor
        self.llm = llm
        self.tools = list(tools or [])
    def call(self, prompt):                     # instance method — uses self.llm
        return self.llm.predict_and_call(self.tools, prompt)
    @classmethod
    def for_document(cls, llm, file_path):      # alt constructor
        return cls(llm, DocumentTools(file_path).as_tools())
    @staticmethod
    def is_pdf(path):                           # utility — no self, no cls
        return path.endswith(".pdf")
```

**Rule of thumb:** a helper whose whole job is "make one of these from friendlier
inputs" is an alternative constructor → make it a `@classmethod` using `cls`.

---

## Python concept: `@property`

### What it is
`@property` turns a **method** into something accessed like an **attribute** — no
parentheses. It runs code *when a field is read*, while keeping `obj.thing`
syntax.

```python
class Circle:
    def __init__(self, radius):
        self.radius = radius
    @property
    def area(self):
        return 3.14159 * self.radius ** 2

c = Circle(10)
c.area        # 314.159   ← no parentheses
c.area()      # TypeError  ← not called like a method
```

### Codebase example — lazy caching in `DocumentIndexer` (indexing.py:37)
```python
class DocumentIndexer:
    def __init__(self, file_path, chunk_size=1024):
        self.file_path = str(file_path)
        self._vector_index = None          # private backing field, starts empty
    @property
    def vector_index(self):
        if self._vector_index is None:     # build only on first access
            from llama_index.core import VectorStoreIndex
            self._vector_index = VectorStoreIndex(self.nodes)
        return self._vector_index          # cached from then on
```
Three things hide behind a plain `indexer.vector_index`:
1. **Laziness** — expensive work (embedding via the API) runs on *first read*, not
   at construction. That's why `RouterEngine(...)` is instant.
2. **Caching** — stored in `self._vector_index`; second access is free. Document
   embedded at most once.
3. **Clean syntax** — `self.indexer.vector_index`, not `get_vector_index()`.

Pattern: private field `_vector_index` (storage) + public property `vector_index`
(controlled access). `nodes`, `summary_index`, and `RouterEngine.engine`
(router_engine.py:48) all follow this identical lazy-cache shape.

### Why use it
1. **Computed/derived values that look like data** — always in sync:
   ```python
   @property
   def full_name(self):
       return f"{self.first} {self.last}"
   ```
2. **Lazy + cached expensive work** (this codebase's use).
3. **Adding logic without breaking callers** — the classic reason. Ship
   `obj.temperature` as a plain attribute, later add validation by converting it
   to a property; existing `obj.temperature` code keeps working. A `@x.setter`
   companion intercepts writes:
   ```python
   class Thermostat:
       def __init__(self): self._temp = 20
       @property
       def temp(self): return self._temp
       @temp.setter
       def temp(self, value):
           if value < 0: raise ValueError("too cold")
           self._temp = value
   ```
   A getter-only property (like all of `DocumentIndexer`'s) is effectively
   **read-only** — no setter means `indexer.vector_index = x` raises. That's a
   feature: nobody can swap the cached index from outside.

### `@property` vs `@classmethod`
| | receives | called on | purpose |
|---|---|---|---|
| normal method | `self` | instance, with `()` | acts on one object's data |
| `@property` | `self` | instance, **no `()`** | access computed/managed data *as a field* |
| `@classmethod` | `cls` | the class | alternative constructor / class-level work |

### When to use it
- **Use** it when you want attribute-style access but need code to run on read —
  computed values, lazy loading, caching, read-only/validated fields.
- **Don't** use it for genuinely expensive or side-effecting actions the caller
  should know are costly — keep those explicit methods with `()`.

**Rule of thumb:** feels like data and is cheap-or-cached → `@property`. *Does*
something (action, request, mutation) → method with parentheses.

---

## Python concept: why `super().__init__(agent)` comes last

Selected line: `multi_document_agent.py:66`.

### The dependency
`super().__init__(agent)` can't run until `agent` exists — and building `agent`
is the whole body of `__init__`:
```python
def __init__(self, file_paths, llm, ...):
    self.file_paths = [...]
    self.tools = []
    for path in self.file_paths:              # 1. gather tools from each PDF
        self.tools.extend(DocumentTools(...).as_tools())
    if use_tool_retrieval is None:            # 2. decide the mode
        use_tool_retrieval = len(self.file_paths) > 3
    if use_tool_retrieval:                    # 3. build the FunctionAgent
        agent = FunctionAgent(tool_retriever=..., ...)
    else:
        agent = FunctionAgent(tools=self.tools, ...)
    super().__init__(agent)                   # 4. hand the finished agent up
```
The base needs a **fully-built `agent`** as its argument, but `agent` is the
*product* of steps 1–3. Calling `super().__init__(agent)` at the top would raise
`NameError` — the variable doesn't exist yet. The base class (agent_runtime.py:32)
is a thin wrapper that just stores what it's given:
```python
class WorkflowAgent:
    def __init__(self, agent):
        self._agent = agent
        self._ctx = None
```

### The general rule
"Call `super().__init__()` first" is a **convention, not a law**. It applies when
the parent sets up state you then build *on top of*. Here it's the opposite: the
parent needs something *you* produce, so the call comes after.
```python
# Parent initializes the foundation you extend → super() FIRST
def __init__(self, ...):
    super().__init__(...)          # parent sets self.x
    self.z = self.x + something    # you build on it

# Parent consumes something you must build → super() LAST  (this case)
def __init__(self, ...):
    agent = ...build it...
    super().__init__(agent)
```

### Same pattern in the sibling class
`ReasoningAgent` (reasoning_agent.py:27) does the identical thing for the same
reason — build the `FunctionAgent` first, hand it up last.

### One caveat
Because `super().__init__` runs last, `self._agent`/`self._ctx` don't exist until
the end of `__init__`. That's fine here — nothing before that line calls
`self.query()` or touches `self._agent`. It's exactly the reason the "super first"
convention exists: if a method called mid-`__init__` depended on base-class state,
that state wouldn't be set up yet. Here the subclass only touches its *own* fields
first, so there's no hazard.

**Bottom line:** the order follows the data dependency, not habit. The parent's
argument is the thing this constructor builds, so you build it first and call
`super().__init__` last.
