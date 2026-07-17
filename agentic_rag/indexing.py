"""Document loading and index construction.

Every lesson starts the same way: read a PDF, split it into ~1024-token nodes,
then build a vector index (for retrieval) and/or a summary index (for
whole-document questions). ``DocumentIndexer`` captures that once and lazily
caches each artifact so a document is parsed and embedded at most once.
"""

from __future__ import annotations


class DocumentIndexer:
    """Load a single document and expose its nodes, vector index, and summary index.

    All three are built on first access and cached, so ``vector_index`` and
    ``summary_index`` share the same parsed nodes rather than re-reading the PDF.
    """

    def __init__(self, file_path: str, chunk_size: int = 1024):
        self.file_path = str(file_path)
        self.chunk_size = chunk_size
        self._nodes = None
        self._vector_index = None
        self._summary_index = None

    @property
    def nodes(self):
        if self._nodes is None:
            from llama_index.core import SimpleDirectoryReader
            from llama_index.core.node_parser import SentenceSplitter

            documents = SimpleDirectoryReader(input_files=[self.file_path]).load_data()
            splitter = SentenceSplitter(chunk_size=self.chunk_size)
            self._nodes = splitter.get_nodes_from_documents(documents)
        return self._nodes

    @property
    def vector_index(self):
        if self._vector_index is None:
            from llama_index.core import VectorStoreIndex

            self._vector_index = VectorStoreIndex(self.nodes)
        return self._vector_index

    @property
    def summary_index(self):
        if self._summary_index is None:
            from llama_index.core import SummaryIndex

            self._summary_index = SummaryIndex(self.nodes)
        return self._summary_index
