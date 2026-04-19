"""
RAG Pipeline Evaluation Script

Generates test documents with known facts, ingests them through the full pipeline,
asks questions with known answers, and scores how well the model performs.

Requires Ollama running with nomic-embed-text and llama3.2 models.

Usage:
    python eval/evaluate_rag.py
"""

import shutil
import tempfile
import time
from dataclasses import dataclass

from langchain_core.documents import Document

from src.splitter import split_documents
from src.embedding import embed_documents
from src.qa_chain import ask_question


# ---------------------------------------------------------------------------
# Test case definition
# ---------------------------------------------------------------------------

@dataclass
class TestCase:
    """A single evaluation case with a question and expected keywords."""
    category: str           # Type of test (factual, numerical, etc.)
    document_text: str      # The source document content
    question: str           # The question to ask the RAG pipeline
    expected_keywords: list # Keywords that should appear in a correct answer


# ---------------------------------------------------------------------------
# Test cases — each has a document, a question, and expected answer keywords
# ---------------------------------------------------------------------------

TEST_CASES = [
    # --- Factual recall ---
    TestCase(
        category="factual",
        document_text=(
            "Aurora Technologies was founded in 2019 by Dr. Elena Marquez in Austin, Texas. "
            "The company specializes in quantum computing hardware and has 340 employees. "
            "Their flagship product is the QuantumCore Q7 processor."
        ),
        question="Who founded Aurora Technologies and where is it located?",
        expected_keywords=["elena marquez", "austin"],
    ),
    TestCase(
        category="factual",
        document_text=(
            "The Velaris Protocol is a network security framework developed by CyberShield Inc. "
            "It uses triple-layer encryption with AES-256, RSA-4096, and ChaCha20. "
            "The protocol was first deployed in March 2023 across 14 government agencies."
        ),
        question="What encryption methods does the Velaris Protocol use?",
        expected_keywords=["aes-256", "rsa-4096", "chacha20"],
    ),

    # --- Numerical extraction ---
    TestCase(
        category="numerical",
        document_text=(
            "Quarterly Financial Report - Q3 2025\n"
            "Total Revenue: $4.8 million\n"
            "Operating Expenses: $2.1 million\n"
            "Net Profit: $1.3 million\n"
            "Employee Count: 87\n"
            "Customer Retention Rate: 94.2%"
        ),
        question="What was the net profit and customer retention rate in Q3 2025?",
        expected_keywords=["1.3", "94"],
    ),
    TestCase(
        category="numerical",
        document_text=(
            "Patient discharge summary: Patient ID 7782. Admitted on 2025-01-15. "
            "Blood pressure at admission: 142/91 mmHg. Heart rate: 88 bpm. "
            "Prescribed Lisinopril 10mg daily. Discharged on 2025-01-18 in stable condition."
        ),
        question="What was the patient's blood pressure and what medication was prescribed?",
        expected_keywords=["142", "lisinopril"],
    ),

    # --- Multi-fact reasoning ---
    TestCase(
        category="multi-fact",
        document_text=(
            "Project Orion Timeline:\n"
            "Phase 1 (Jan-Mar 2025): Requirements gathering. Lead: Sarah Chen. Budget: $50,000.\n"
            "Phase 2 (Apr-Jun 2025): Development. Lead: Marcus Webb. Budget: $120,000.\n"
            "Phase 3 (Jul-Sep 2025): Testing. Lead: Sarah Chen. Budget: $35,000.\n"
            "Total project budget: $205,000."
        ),
        question="Who led the development phase and what was its budget?",
        expected_keywords=["marcus webb", "120,000"],
    ),

    # --- Negation / refusal test ---
    TestCase(
        category="refusal",
        document_text=(
            "The cafeteria menu for Monday includes grilled chicken, Caesar salad, "
            "tomato soup, and sparkling water. Dessert options are chocolate cake and fruit cups."
        ),
        question="What is the cafeteria menu for Friday?",
        expected_keywords=["don't have enough context", "not", "no information", "friday"],
    ),

    # --- Long document with buried detail ---
    TestCase(
        category="needle-in-haystack",
        document_text=(
            "Annual Report 2025 - Greenfield Energy Corp.\n\n"
            "Section 1: Company Overview\n"
            "Greenfield Energy is a renewable energy provider operating across 12 states. "
            "Founded in 2010, the company has grown from a small startup to a major player "
            "in the solar and wind energy sectors.\n\n"
            "Section 2: Operations\n"
            "The company operates 47 solar farms and 23 wind installations. Their largest "
            "solar farm is located in Barstow, California, producing 340 MW annually.\n\n"
            "Section 3: Financial Highlights\n"
            "Revenue increased 18% year-over-year to $890 million. The company invested "
            "$145 million in R&D, focusing on next-generation perovskite solar cells.\n\n"
            "Section 4: Workforce\n"
            "Total employees: 3,200. The engineering team expanded by 22% this year. "
            "The company's chief technology officer, Dr. James Okafor, announced plans "
            "to open a new research lab in Boulder, Colorado by Q2 2026.\n\n"
            "Section 5: Sustainability Goals\n"
            "Greenfield aims to achieve carbon neutrality by 2028 and plans to double "
            "its wind energy capacity over the next three years."
        ),
        question="Where is Greenfield Energy opening a new research lab and who announced it?",
        expected_keywords=["boulder", "okafor"],
    ),

    # --- Technical jargon ---
    TestCase(
        category="technical",
        document_text=(
            "The microservice uses a CQRS pattern with event sourcing. Write operations "
            "go through a Kafka topic (orders.commands) and are processed by the command handler. "
            "Read operations query a denormalized PostgreSQL view (orders_read_model). "
            "The event store uses Apache Cassandra with a TTL of 90 days."
        ),
        question="What database is used for the event store and what is the TTL?",
        expected_keywords=["cassandra", "90"],
    ),
]


# ---------------------------------------------------------------------------
# Evaluation logic
# ---------------------------------------------------------------------------

def score_response(response: str, expected_keywords: list) -> dict:
    """Check how many expected keywords appear in the model's response.

    Args:
        response: The LLM's answer string.
        expected_keywords: Keywords that should be present in a correct answer.

    Returns:
        A dict with hits, misses, and the score as a percentage.
    """
    response_lower = response.lower()
    hits = [kw for kw in expected_keywords if kw.lower() in response_lower]
    misses = [kw for kw in expected_keywords if kw.lower() not in response_lower]

    score = len(hits) / len(expected_keywords) * 100 if expected_keywords else 0

    return {"hits": hits, "misses": misses, "score": score}


def run_evaluation():
    """Run all test cases through the full RAG pipeline and print a scorecard."""
    # Create a temporary directory for the test ChromaDB
    tmp_db = tempfile.mkdtemp()

    print("=" * 70)
    print("CORTEX ENGINE — RAG EVALUATION")
    print("=" * 70)

    try:
        # Step 1: Build documents from all test cases
        print("\n[1/3] Preparing test documents...")
        all_docs = []
        for i, tc in enumerate(TEST_CASES):
            all_docs.append(Document(
                page_content=tc.document_text,
                metadata={"source": f"test_case_{i}", "category": tc.category},
            ))

        # Step 2: Split and embed all documents into a temp ChromaDB
        print("[2/3] Splitting and embedding into temporary database...")
        chunks = split_documents(all_docs)
        embed_documents(chunks, persist_directory=tmp_db)
        print(f"       Embedded {len(all_docs)} documents as {len(chunks)} chunks.\n")

        # Step 3: Run each question through the QA chain and score
        print("[3/3] Running evaluation...\n")
        print("-" * 70)

        results = []
        for i, tc in enumerate(TEST_CASES):
            print(f"\nTest {i+1}/{len(TEST_CASES)} [{tc.category}]")
            print(f"  Q: {tc.question}")

            start = time.time()
            response = ask_question(tc.question, persist_directory=tmp_db)
            elapsed = time.time() - start

            result = score_response(response, tc.expected_keywords)
            results.append(result)

            print(f"  A: {response[:200]}{'...' if len(response) > 200 else ''}")
            print(f"  Expected keywords: {tc.expected_keywords}")
            print(f"  Hits: {result['hits']}  |  Misses: {result['misses']}")
            print(f"  Score: {result['score']:.0f}%  |  Time: {elapsed:.1f}s")
            print("-" * 70)

        # --- Summary ---
        total_score = sum(r["score"] for r in results) / len(results)
        perfect = sum(1 for r in results if r["score"] == 100)

        # Per-category breakdown
        categories = {}
        for tc, r in zip(TEST_CASES, results):
            if tc.category not in categories:
                categories[tc.category] = []
            categories[tc.category].append(r["score"])

        print("\n" + "=" * 70)
        print("SCORECARD")
        print("=" * 70)
        print(f"  Overall accuracy:    {total_score:.1f}%")
        print(f"  Perfect answers:     {perfect}/{len(results)}")
        print(f"  Total test cases:    {len(results)}")
        print()
        print("  Per-category breakdown:")
        for cat, scores in categories.items():
            avg = sum(scores) / len(scores)
            print(f"    {cat:<20} {avg:.0f}%  ({len(scores)} tests)")
        print("=" * 70)

    finally:
        # Clean up the temporary database
        shutil.rmtree(tmp_db, ignore_errors=True)


if __name__ == "__main__":
    run_evaluation()
