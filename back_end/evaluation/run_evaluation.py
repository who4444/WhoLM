"""
WhoLM RAG Evaluation Runner

Run the full evaluation suite against the RAG pipeline.
Compares dense-only vs hybrid retrieval and generates detailed reports.

Usage:
    cd back_end
    python -m evaluation.run_evaluation                     # Full eval
    python -m evaluation.run_evaluation --dry-run           # Test pipeline connection
    python -m evaluation.run_evaluation --mode retrieval    # Retrieval metrics only
    python -m evaluation.run_evaluation --mode generation   # Generation metrics only
    python -m evaluation.run_evaluation --compare           # Compare dense vs hybrid
"""

import sys
import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

# Ensure back_end is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.retrieval_metrics import RetrievalEvaluator
from evaluation.generation_metrics import GenerationEvaluator

logger = logging.getLogger(__name__)

# Default paths
EVAL_DATASET_PATH = Path(__file__).parent / "eval_dataset.json"
RESULTS_DIR = Path(__file__).parent / "results"


def load_eval_dataset(path: str = None) -> List[Dict[str, Any]]:
    """Load the evaluation dataset from JSON."""
    dataset_path = Path(path) if path else EVAL_DATASET_PATH

    if not dataset_path.exists():
        logger.error(f"Evaluation dataset not found: {dataset_path}")
        logger.info("Create eval_dataset.json with your test queries and relevant doc IDs.")
        return []

    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    logger.info(f"Loaded {len(dataset)} evaluation queries from {dataset_path}")
    return dataset


def get_rag_pipeline(use_hybrid: bool = True):
    """Initialize the RAG pipeline for evaluation."""
    try:
        from services.rag.qdrant_rag_pipeline import QdrantRAGPipeline
        from config.config import Config

        pipeline = QdrantRAGPipeline(
            qdrant_url=Config.QDRANT_URL,
            bm25_persist_path="bm25_data/bm25_index.pkl"
        )
        logger.info(f"RAG pipeline initialized (BM25 docs: {pipeline.hybrid_retriever.bm25_doc_count})")
        return pipeline
    except Exception as e:
        logger.error(f"Failed to initialize RAG pipeline: {e}")
        raise


def run_retrieval_evaluation(pipeline, dataset: List[Dict], use_hybrid: bool = True,
                             k_values: List[int] = None) -> Dict[str, Any]:
    """
    Run retrieval evaluation on the dataset.

    Args:
        pipeline: QdrantRAGPipeline instance
        dataset: List of evaluation queries with relevant_doc_ids
        use_hybrid: Whether to use hybrid retrieval
        k_values: K values for metrics

    Returns:
        Dict with evaluation results
    """
    evaluator = RetrievalEvaluator(k_values=k_values or [3, 5, 10])

    # Filter to queries that have relevant_doc_ids defined
    valid_queries = [q for q in dataset if q.get("relevant_doc_ids")]
    
    if not valid_queries:
        logger.warning("No queries with relevant_doc_ids found in dataset. "
                      "Running retrieval without ground truth (will only report counts).")
        # Still run queries to verify pipeline works
        for query_data in dataset:
            question = query_data["question"]
            try:
                results = pipeline.query(question, use_hybrid=use_hybrid)
                retrieved_ids = [r.get("doc_id", "") for r in results]
                logger.info(f"  Query: '{question[:60]}...' → {len(retrieved_ids)} results")
            except Exception as e:
                logger.error(f"  Query failed: '{question[:60]}...' → {e}")

        return {
            "mode": "hybrid" if use_hybrid else "dense",
            "queries_evaluated": 0,
            "note": "No ground truth doc IDs provided. Add relevant_doc_ids to eval_dataset.json.",
            "pipeline_functional": True
        }

    evaluated = 0
    for query_data in valid_queries:
        question = query_data["question"]
        relevant_ids = set(query_data["relevant_doc_ids"])

        try:
            results = pipeline.query(question, use_hybrid=use_hybrid)
            retrieved_ids = [r.get("doc_id", "") for r in results]

            per_query = evaluator.add_result(retrieved_ids, relevant_ids, query=question)
            evaluated += 1

            logger.info(f"  [{evaluated}/{len(valid_queries)}] '{question[:50]}...' "
                       f"→ P@5={per_query.get('precision@5', 0):.3f}, "
                       f"MRR={per_query.get('mrr', 0):.3f}")
        except Exception as e:
            logger.error(f"  Query failed: '{question[:50]}...' → {e}")

    metrics = evaluator.compute_metrics()
    metrics["mode"] = "hybrid" if use_hybrid else "dense"
    metrics["summary"] = evaluator.summary_table()
    metrics["per_query"] = evaluator.get_per_query_results()

    return metrics


def run_generation_evaluation(pipeline, dataset: List[Dict],
                              use_llm_judge: bool = True) -> Dict[str, Any]:
    """
    Run generation evaluation on the dataset.

    Args:
        pipeline: QdrantRAGPipeline instance
        dataset: Evaluation queries
        use_llm_judge: Whether to use LLM-as-judge

    Returns:
        Dict with evaluation results
    """
    evaluator = GenerationEvaluator(use_llm_judge=use_llm_judge)

    for i, query_data in enumerate(dataset):
        question = query_data["question"]

        try:
            # Get retrieval results
            results = pipeline.query(question, use_hybrid=True)

            # Build context from results
            context_parts = [r.get("text", "") for r in results if r.get("text")]
            context = "\n---\n".join(context_parts)

            # Generate answer using the chatbot's LLM
            try:
                from services.chatbot.gemini_client import generate_response
                from services.chatbot.prompts.document_prompts import DOC_PROMPT

                answer = generate_response(
                    sys_prompt=DOC_PROMPT,
                    context=context,
                    query=question
                )
            except Exception as e:
                logger.warning(f"LLM generation failed, using context as answer: {e}")
                answer = context[:500] if context else "No answer generated"

            per_query = evaluator.add_result(
                question=question,
                answer=answer,
                context=context,
                expected_answer=query_data.get("expected_answer")
            )

            logger.info(f"  [{i+1}/{len(dataset)}] '{question[:50]}...' "
                       f"→ Faith={per_query['faithfulness_score']:.3f}, "
                       f"Rel={per_query['relevance_score']:.3f}")

        except Exception as e:
            logger.error(f"  Generation eval failed: '{question[:50]}...' → {e}")

    metrics = evaluator.compute_metrics()
    metrics["summary"] = evaluator.summary_table()
    metrics["per_query"] = evaluator.get_per_query_results()

    return metrics


def run_comparison(pipeline, dataset: List[Dict], k_values: List[int] = None) -> Dict[str, Any]:
    """
    Compare dense-only vs hybrid retrieval side-by-side.

    Returns:
        Dict with both result sets and comparison
    """
    print("\n" + "=" * 60)
    print("  DENSE-ONLY RETRIEVAL")
    print("=" * 60)
    dense_results = run_retrieval_evaluation(pipeline, dataset, use_hybrid=False, k_values=k_values)

    print("\n" + "=" * 60)
    print("  HYBRID RETRIEVAL (BM25 + Dense)")
    print("=" * 60)
    hybrid_results = run_retrieval_evaluation(pipeline, dataset, use_hybrid=True, k_values=k_values)

    # Build comparison
    comparison = {
        "dense": dense_results,
        "hybrid": hybrid_results,
    }

    # Compute deltas if both have metrics
    if dense_results.get("num_queries") and hybrid_results.get("num_queries"):
        deltas = {}
        for key in dense_results:
            if key.startswith("avg_") and key in hybrid_results:
                delta = hybrid_results[key] - dense_results[key]
                deltas[key] = {
                    "dense": dense_results[key],
                    "hybrid": hybrid_results[key],
                    "delta": delta,
                    "improved": delta > 0
                }
        comparison["deltas"] = deltas

        # Print comparison table
        print("\n" + "=" * 60)
        print("  COMPARISON: Dense vs Hybrid")
        print("=" * 60)
        print(f"  {'Metric':<25} {'Dense':>8} {'Hybrid':>8} {'Delta':>8}")
        print("-" * 60)
        for metric, values in deltas.items():
            name = metric.replace("avg_", "")
            arrow = "↑" if values["improved"] else "↓" if values["delta"] < 0 else "="
            print(f"  {name:<25} {values['dense']:>8.4f} {values['hybrid']:>8.4f} "
                  f"{values['delta']:>+8.4f} {arrow}")
        print("=" * 60)

    return comparison


def save_results(results: Dict[str, Any], name: str):
    """Save evaluation results to JSON file."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{name}_{timestamp}.json"
    filepath = RESULTS_DIR / filename

    # Remove non-serializable items
    clean_results = {}
    for k, v in results.items():
        if k == "summary":
            clean_results[k] = v  # string
        elif k == "per_query":
            clean_results[k] = v  # list of dicts
        elif isinstance(v, (int, float, str, bool, list, dict, type(None))):
            clean_results[k] = v

    with open(filepath, "w") as f:
        json.dump(clean_results, f, indent=2, default=str)

    logger.info(f"Results saved to {filepath}")
    return filepath


def main():
    parser = argparse.ArgumentParser(
        description="WhoLM RAG Evaluation Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m evaluation.run_evaluation                     # Full evaluation
  python -m evaluation.run_evaluation --dry-run           # Test connection
  python -m evaluation.run_evaluation --mode retrieval    # Retrieval only
  python -m evaluation.run_evaluation --compare           # Dense vs Hybrid
  python -m evaluation.run_evaluation --dataset my_eval.json  # Custom dataset
        """
    )

    parser.add_argument("--mode", choices=["retrieval", "generation", "full"],
                       default="full", help="Evaluation mode")
    parser.add_argument("--dataset", type=str, default=None,
                       help="Path to evaluation dataset JSON")
    parser.add_argument("--compare", action="store_true",
                       help="Compare dense-only vs hybrid retrieval")
    parser.add_argument("--dry-run", action="store_true",
                       help="Test pipeline connectivity without full evaluation")
    parser.add_argument("--no-llm-judge", action="store_true",
                       help="Skip LLM-as-judge, use heuristic scoring")
    parser.add_argument("--save", action="store_true",
                       help="Save results to evaluation/results/")
    parser.add_argument("--k-values", type=str, default="3,5,10",
                       help="Comma-separated K values for metrics (default: 3,5,10)")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler()]
    )

    k_values = [int(k.strip()) for k in args.k_values.split(",")]

    print("\n" + "=" * 60)
    print("  WhoLM RAG Evaluation Runner")
    print("=" * 60)

    # Load dataset
    dataset = load_eval_dataset(args.dataset)
    if not dataset:
        print("\n❌ No evaluation dataset. Create evaluation/eval_dataset.json first.")
        sys.exit(1)

    # Initialize pipeline
    print("\nInitializing RAG pipeline...")
    try:
        pipeline = get_rag_pipeline()
    except Exception as e:
        print(f"\n❌ Failed to initialize pipeline: {e}")
        sys.exit(1)

    # Dry run
    if args.dry_run:
        print("\n🔍 Dry run: testing pipeline connectivity...")
        try:
            stats = pipeline.get_stats()
            print(f"  ✅ Pipeline connected")
            print(f"  Text docs in Qdrant: {stats['total_text_documents']}")
            print(f"  Frame docs in Qdrant: {stats['total_frame_documents']}")
            print(f"  BM25 index size: {stats['bm25_index_size']}")
            print(f"  BM25 ready: {stats['bm25_ready']}")
            print(f"  Hybrid weights: {stats['hybrid_weights']}")

            # Test a single query
            test_q = dataset[0]["question"]
            print(f"\n  Testing query: '{test_q[:60]}...'")
            results = pipeline.query(test_q, retriever_top_k=3)
            print(f"  ✅ Got {len(results)} results")
            for r in results[:3]:
                print(f"    - [{r.get('collection_type', '?')}] "
                      f"score={r.get('score', 0):.4f} "
                      f"text='{r.get('text', '')[:60]}...'")

            print("\n✅ Dry run successful!")
        except Exception as e:
            print(f"\n❌ Dry run failed: {e}")
            sys.exit(1)
        return

    # Run comparison
    if args.compare:
        print("\nRunning dense vs hybrid comparison...")
        results = run_comparison(pipeline, dataset, k_values=k_values)
        if args.save:
            save_results(results, "comparison")
        return

    # Run evaluation
    all_results = {}

    if args.mode in ["retrieval", "full"]:
        print("\n📊 Running retrieval evaluation...")
        retrieval_results = run_retrieval_evaluation(
            pipeline, dataset, use_hybrid=True, k_values=k_values
        )
        if retrieval_results.get("summary"):
            print("\n" + retrieval_results["summary"])
        all_results["retrieval"] = retrieval_results

    if args.mode in ["generation", "full"]:
        print("\n📊 Running generation evaluation...")
        gen_results = run_generation_evaluation(
            pipeline, dataset, use_llm_judge=not args.no_llm_judge
        )
        if gen_results.get("summary"):
            print("\n" + gen_results["summary"])
        all_results["generation"] = gen_results

    # Save results
    if args.save:
        save_results(all_results, args.mode)

    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()
