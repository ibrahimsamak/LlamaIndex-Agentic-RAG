"""Environment and model configuration.

Centralizes API-key loading, the async patch every LlamaIndex query engine
in this project relies on, and the LLM / embedding-model factory. Swapping
provider or model happens here and nowhere else.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


def load_environment() -> str:
    """Load ``.env``, apply the ``nest_asyncio`` patch, and return the OpenAI key.

    The summary query engines are built with ``use_async=True``; ``nest_asyncio``
    lets them run inside an already-running event loop (notebooks, REPLs, some
    CLIs). Raises if ``OPENAI_API_KEY`` is missing so failures are obvious.
    """
    load_dotenv()

    import nest_asyncio

    nest_asyncio.apply()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to a .env file or the environment."
        )
    return api_key


@dataclass
class RAGConfig:
    """Model choices shared across every engine and agent in the project.

    Kept as plain data plus two factory methods so the rest of the codebase
    never imports a provider SDK directly — change these fields (or subclass)
    to target a different model.
    """

    llm_model: str = "gpt-3.5-turbo"
    embed_model: str = "text-embedding-ada-002"
    chunk_size: int = 1024
    temperature: float = 0.0

    def build_llm(self):
        from llama_index.llms.openai import OpenAI

        return OpenAI(model=self.llm_model, temperature=self.temperature)

    def build_embed_model(self):
        from llama_index.embeddings.openai import OpenAIEmbedding

        return OpenAIEmbedding(model=self.embed_model)

    def apply_global_settings(self) -> None:
        """Install this config's LLM and embedding model as LlamaIndex globals.

        Engines that don't take an explicit ``llm``/``embed_model`` (e.g. the
        router in Lesson 1) pick these up from ``Settings``.
        """
        from llama_index.core import Settings

        Settings.llm = self.build_llm()
        Settings.embed_model = self.build_embed_model()
