"""RAGAS evaluation pipeline for a simple Kubernetes-docs RAG system.

Builds a FAISS RAG over a handful of Kubernetes documentation pages, generates a
synthetic test set with RAGAS, answers each question with the RAG, and scores the
answers with a range of RAGAS metrics (ROUGE, LLM-judged, and RAG-specific).

Run:  python3 ragas_eval.py
Requires OPENAI_API_KEY in .env.

NOTE: This file is intentionally NOT named ragas.py — that would shadow the
installed `ragas` library (`import ragas` would import this file instead).
"""

import asyncio
import os
import ssl

import numpy as np
import nltk
from dotenv import load_dotenv
from openai import OpenAI

# RAG stack
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS

# RAGAS — modern API for evaluation LLM/embeddings.
from ragas.llms import llm_factory
from ragas.embeddings import OpenAIEmbeddings as RagasOpenAIEmbeddings

# RAGAS — legacy wrappers are still required by TestsetGenerator, which only
# accepts a BaseRagasLLM (the modern llm_factory object is not accepted there).
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset import TestsetGenerator
from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.transforms import apply_transforms, default_transforms_for_prechunked
from ragas.run_config import RunConfig

# Bound RAGAS concurrency. The default (max_workers=16) opens many simultaneous
# connections, which is the usual trigger for `openai.APIConnectionError` even
# when individual calls succeed. Fewer workers + generous retries is far more
# reliable at the cost of some speed.
RAGAS_RUN_CONFIG = RunConfig(max_workers=4, timeout=180, max_retries=10)

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import (
    RougeScore,
    SimpleCriteriaScore,
    RubricsScore,
    SemanticSimilarity,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ResponseRelevancy,
)
from ragas.metrics._factual_correctness import FactualCorrectness

# urls for kubernetes documentation
urls = [
    "https://kubernetes.io/docs/concepts/overview/",
    "https://kubernetes.io/docs/concepts/architecture/",
    "https://kubernetes.io/docs/concepts/containers/",
    "https://kubernetes.io/docs/concepts/workloads/",
    "https://kubernetes.io/docs/concepts/storage/",
]

# Reuse a single event loop for all async RAGAS scorers so the shared async HTTP
# clients aren't torn down between calls.
_LOOP = asyncio.new_event_loop()


def ascore(scorer, sample):
    """Run an async single-turn RAGAS scorer synchronously and return the score."""
    return _LOOP.run_until_complete(scorer.single_turn_ascore(sample))


def setup_nltk():
    """Download NLTK data, working around macOS SSL cert verification failures."""
    ssl._create_default_https_context = ssl._create_unverified_context
    nltk.download("punkt_tab")
    nltk.download("averaged_perceptron_tagger_eng")


def build_rag():
    """Load + split the Kubernetes docs, embed them, and return (db, documents)."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=300,
    )
    loader = UnstructuredURLLoader(urls=urls)
    documents = loader.load_and_split(text_splitter)

    # Embed the chunks
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(documents, embeddings)
    return db, documents


def rag(db, query, k=5):
    """Retrieve context from `db` and answer `query`. Returns (answer, contexts)."""
    # Retrieving the context
    retrieved_docs = db.similarity_search(query, k=k)

    # Combine all the context from the retrieved docs
    combined_text = "\n\n".join([doc.page_content for doc in retrieved_docs])

    # Define the prompt
    prompt = f""" based on this context: {combined_text} answer the query: {query}"""

    # Call the LLM model (retries/timeout guard against transient connection errors)
    model = ChatOpenAI(model="gpt-4.1-mini", temperature=0, max_retries=6, timeout=60)
    response_text = model.invoke(prompt)

    return response_text.content, [txt.page_content for txt in retrieved_docs]


def generate_testset(documents, testset_size=30):
    """Generate a synthetic RAGAS test set from the already-chunked documents.

    TestsetGenerator only accepts the legacy BaseRagasLLM/BaseRagasEmbeddings, so
    we wrap the LangChain models here (unlike the metrics, which use llm_factory).

    We do NOT use generator.generate_with_langchain_docs(): it builds DOCUMENT
    nodes and runs the default transforms, whose HeadlineSplitter hard-fails with
    "'headlines' property not found in this node" on nodes the extractor left
    without headlines. Since `documents` is already chunked (by the RAG's text
    splitter), we build a CHUNK-node knowledge graph and apply the prechunked
    transforms, which skip the headline extract/split step entirely.
    """
    generator_llm = LangchainLLMWrapper(
        ChatOpenAI(model="gpt-4.1-mini", temperature=0, max_retries=6, timeout=60)
    )
    generator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(max_retries=6))

    # Build a knowledge graph of CHUNK nodes from the pre-split documents.
    kg = KnowledgeGraph()
    for doc in documents:
        kg.nodes.append(
            Node(
                type=NodeType.CHUNK,
                properties={
                    "page_content": doc.page_content,
                    "document_metadata": doc.metadata,
                },
            )
        )

    # Enrich the graph (summaries, themes, NER, similarity relationships) without
    # the headline splitter that breaks on these docs.
    transforms = default_transforms_for_prechunked(
        llm=generator_llm, embedding_model=generator_embeddings
    )
    apply_transforms(kg, transforms, run_config=RAGAS_RUN_CONFIG)

    generator = TestsetGenerator(
        llm=generator_llm,
        embedding_model=generator_embeddings,
        knowledge_graph=kg,
    )
    dataset = generator.generate(
        testset_size=testset_size, run_config=RAGAS_RUN_CONFIG
    )
    return dataset.to_pandas()


def answer_testset(db, test_df):
    """Answer each user_input with the RAG and attach answer/context columns."""
    answers_gen = []
    context_gen = []
    for user_input in test_df["user_input"]:
        answer, context = rag(db, user_input)
        answers_gen.append(answer)
        context_gen.append(context)

    test_df["answer"] = answers_gen
    test_df["context"] = context_gen
    return test_df


def evaluate_rouge(test_df):
    """ROUGE score — traditional overlap-based metric (no LLM)."""
    scorer = RougeScore()
    scores = []
    for row in test_df.to_dict(orient="records"):
        sample = SingleTurnSample(
            user_input=row["user_input"],
            reference=row["reference"],
            response=row["answer"],
        )
        scores.append(ascore(scorer, sample))
    print(scores)
    print(f"The mean rouge score is {np.mean(scores)}")


def evaluate_simple(eval_llm, test_df):
    """Simple criteria scoring (0-5 similarity), LLM-judged."""
    simple_scorer = SimpleCriteriaScore(
        name="simple scorer",
        definition="Score 0 to 5 by similarity",
        llm=eval_llm,
    )
    scores = []
    for row in test_df.to_dict(orient="records"):
        sample = SingleTurnSample(
            user_input=row["user_input"],
            reference=row["reference"],
            response=row["answer"],
        )
        scores.append(ascore(simple_scorer, sample))
    print(scores)
    print(f"The mean simple score is {np.mean(scores)}")


def evaluate_rubrics(eval_llm, test_df):
    """Rubrics-based scoring (1-5), LLM-judged."""
    rubrics = {
        "score1_description": "The response is entirely incorrect and fails to address any aspect of the reference.",
        "score2_description": "The response contains partial accuracy but includes major errors or significant omissions that affect its relevance to the reference.",
        "score3_description": "The response is mostly accurate but lacks clarity, thoroughness, or minor details needed to fully address the reference.",
        "score4_description": "The response is accurate and clear, with only minor omissions or slight inaccuracies in addressing the reference.",
        "score5_description": "The response is completely accurate, clear, and thoroughly addresses the reference without any errors or omissions.",
    }
    rubrics_scorer = RubricsScore(rubrics=rubrics, llm=eval_llm)
    scores = []
    for row in test_df.to_dict(orient="records"):
        sample = SingleTurnSample(
            user_input=row["user_input"],
            reference=row["reference"],
            response=row["answer"],
        )
        scores.append(ascore(rubrics_scorer, sample))
    print(scores)
    print(f"The mean rubrics score is {np.mean(scores)}")


def evaluate_factual(eval_llm, test_df):
    """Factual correctness of the answer vs. the reference."""
    factual_scorer = FactualCorrectness(llm=eval_llm)
    scores = []
    for row in test_df.to_dict(orient="records"):
        sample = SingleTurnSample(
            user_input=row["user_input"],
            reference=row["reference"],
            response=row["answer"],
        )
        scores.append(ascore(factual_scorer, sample))
    print(scores)
    print(f"The mean factual score is {np.mean(scores)}")


def evaluate_semantic(eval_embeddings, test_df):
    """Semantic similarity between answer and reference (embedding-based)."""
    semantic_scorer = SemanticSimilarity(embeddings=eval_embeddings)
    scores = []
    for row in test_df.to_dict(orient="records"):
        sample = SingleTurnSample(
            user_input=row["user_input"],
            reference=row["reference"],
            response=row["answer"],
        )
        scores.append(ascore(semantic_scorer, sample))
    print(scores)
    print(f"The mean semantic score is {np.mean(scores)}")


def evaluate_context_precision(eval_llm, test_df):
    """LLM context precision with reference."""
    context_scorer = LLMContextPrecisionWithReference(llm=eval_llm)
    scores = []
    for row in test_df.to_dict(orient="records"):
        sample = SingleTurnSample(
            user_input=row["user_input"],
            reference=row["reference"],
            response=row["answer"],
            retrieved_contexts=row["context"],
        )
        scores.append(ascore(context_scorer, sample))
    print(scores)
    print(f"The mean context score is {np.mean(scores)}")


def evaluate_context_recall(eval_llm, test_df):
    """LLM context recall."""
    context_recall_scorer = LLMContextRecall(llm=eval_llm)
    scores = []
    for row in test_df.to_dict(orient="records"):
        sample = SingleTurnSample(
            user_input=row["user_input"],
            reference=row["reference"],
            response=row["answer"],
            retrieved_contexts=row["context"],
        )
        scores.append(ascore(context_recall_scorer, sample))
    print(scores)
    print(f"The mean context recall score is {np.mean(scores)}")


def evaluate_response_relevancy(eval_llm, eval_embeddings, test_df):
    """Response relevancy to the user question."""
    response_relevancy_scorer = ResponseRelevancy(
        llm=eval_llm, embeddings=eval_embeddings
    )
    scores = []
    for row in test_df.to_dict(orient="records"):
        sample = SingleTurnSample(
            user_input=row["user_input"],
            reference=row["reference"],
            response=row["answer"],
            retrieved_contexts=row["context"],
        )
        scores.append(ascore(response_relevancy_scorer, sample))
    print(scores)
    print(f"The mean response relevancy score is {np.mean(scores)}")


def main():
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set (add it to your .env).")

    # --- Setup ---
    setup_nltk()

    # --- Data + RAG ---
    print("\n=== Building RAG ===")
    db, documents = build_rag()

    # Quick smoke test of the RAG
    answer, _ = rag(db, "What is the overview of kubernetes?")
    print(answer)

    # --- Generate synthetic test data ---
    print("\n=== Generating synthetic test set ===")
    test_df = generate_testset(documents, testset_size=30)
    print(test_df.head())

    # Answer each question with our RAG
    print("\n=== Answering test set with RAG ===")
    test_df = answer_testset(db, test_df)
    print(test_df.head())

    # --- Evaluation LLM / embeddings (modern ragas API) ---
    client = OpenAI()  # reads OPENAI_API_KEY from the environment
    eval_llm = llm_factory("gpt-4.1", provider="openai", client=client)
    eval_embeddings = RagasOpenAIEmbeddings(
        client=client, model="text-embedding-3-small"
    )

    print("\n=== Rouge Score ===")
    evaluate_rouge(test_df)

    print("\n=== Simple Scoring ===")
    evaluate_simple(eval_llm, test_df)

    print("\n=== Rubrics Scoring ===")
    evaluate_rubrics(eval_llm, test_df)

    # print("\n=== Factual Correctness ===")
    # evaluate_factual(eval_llm, test_df)

    # print("\n=== Semantic Similarity ===")
    # evaluate_semantic(eval_embeddings, test_df)

    # print("\n=== Context Precision ===")
    # evaluate_context_precision(eval_llm, test_df)

    # print("\n=== Context Recall ===")
    # evaluate_context_recall(eval_llm, test_df)

    # print("\n=== Response Relevancy ===")
    # evaluate_response_relevancy(eval_llm, eval_embeddings, test_df)


if __name__ == "__main__":
    try:
        main()
    finally:
        _LOOP.close()
