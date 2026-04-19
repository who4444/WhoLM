"""
Generation evaluation metrics for the WhoLM RAG pipeline.

Uses LLM-as-judge (via Gemini) to evaluate:
  - Faithfulness: Is the answer grounded in the provided context?
  - Answer Relevance: Does the answer address the user's question?
"""

import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


def _get_gemini_judge():
    """Get the Gemini LLM for evaluation (lazy-loaded)."""
    try:
        from services.chatbot.gemini_client import generate_response
        return generate_response
    except ImportError:
        logger.warning("Gemini client not available, using fallback scoring")
        return None


def faithfulness_score(answer: str, context: str, generate_fn=None) -> Dict[str, Any]:
    """
    Evaluate whether the answer is faithfully grounded in the provided context.
    Uses LLM-as-judge to check for hallucinations.

    Args:
        answer: The generated answer
        context: The retrieved context used for generation
        generate_fn: Optional generation function (defaults to Gemini)

    Returns:
        Dict with 'score' (0-1), 'reasoning', and 'verdict'
    """
    if not answer or not context:
        return {"score": 0.0, "reasoning": "Empty answer or context", "verdict": "FAIL"}

    if generate_fn is None:
        generate_fn = _get_gemini_judge()

    if generate_fn is None:
        # Fallback: simple overlap heuristic
        return _heuristic_faithfulness(answer, context)

    judge_prompt = """You are an impartial judge evaluating whether an AI answer is faithfully grounded in the provided context.

EVALUATION CRITERIA:
- Score 1.0: Every claim in the answer is directly supported by the context
- Score 0.75: Most claims are supported, with minor extrapolations that are reasonable
- Score 0.5: Some claims are supported, but some are not found in the context
- Score 0.25: Few claims are supported; significant hallucination
- Score 0.0: The answer contains fabricated information not in the context

Respond in EXACTLY this format (no other text):
SCORE: <number>
VERDICT: <PASS or FAIL>
REASONING: <one sentence explanation>"""

    judge_context = f"""CONTEXT:
{context[:3000]}

ANSWER TO EVALUATE:
{answer[:1500]}"""

    try:
        result = generate_fn(
            sys_prompt=judge_prompt,
            context=judge_context,
            query="Evaluate the faithfulness of this answer."
        )

        return _parse_judge_response(result)

    except Exception as e:
        logger.error(f"Faithfulness evaluation failed: {e}")
        return _heuristic_faithfulness(answer, context)


def answer_relevance_score(answer: str, question: str, generate_fn=None) -> Dict[str, Any]:
    """
    Evaluate whether the answer is relevant to the question.
    Uses LLM-as-judge to score relevance.

    Args:
        answer: The generated answer
        question: The user's original question
        generate_fn: Optional generation function (defaults to Gemini)

    Returns:
        Dict with 'score' (0-1), 'reasoning', and 'verdict'
    """
    if not answer or not question:
        return {"score": 0.0, "reasoning": "Empty answer or question", "verdict": "FAIL"}

    if generate_fn is None:
        generate_fn = _get_gemini_judge()

    if generate_fn is None:
        return {"score": 0.5, "reasoning": "LLM judge not available", "verdict": "UNKNOWN"}

    judge_prompt = """You are an impartial judge evaluating whether an AI answer is relevant to the user's question.

EVALUATION CRITERIA:
- Score 1.0: Directly and completely answers the question
- Score 0.75: Mostly answers the question with minor gaps
- Score 0.5: Partially relevant but misses key aspects
- Score 0.25: Tangentially related but doesn't answer the question
- Score 0.0: Completely irrelevant or off-topic

Respond in EXACTLY this format (no other text):
SCORE: <number>
VERDICT: <PASS or FAIL>
REASONING: <one sentence explanation>"""

    judge_context = f"""QUESTION:
{question}

ANSWER TO EVALUATE:
{answer[:1500]}"""

    try:
        result = generate_fn(
            sys_prompt=judge_prompt,
            context=judge_context,
            query="Evaluate the relevance of this answer to the question."
        )

        return _parse_judge_response(result)

    except Exception as e:
        logger.error(f"Relevance evaluation failed: {e}")
        return {"score": 0.5, "reasoning": f"Evaluation error: {str(e)}", "verdict": "UNKNOWN"}


def _parse_judge_response(response_text: str) -> Dict[str, Any]:
    """Parse the structured judge response into a dict."""
    result = {"score": 0.0, "reasoning": "", "verdict": "UNKNOWN"}

    for line in response_text.strip().split("\n"):
        line = line.strip()
        if line.upper().startswith("SCORE:"):
            try:
                score_str = line.split(":", 1)[1].strip()
                result["score"] = max(0.0, min(1.0, float(score_str)))
            except (ValueError, IndexError):
                pass
        elif line.upper().startswith("VERDICT:"):
            result["verdict"] = line.split(":", 1)[1].strip().upper()
        elif line.upper().startswith("REASONING:"):
            result["reasoning"] = line.split(":", 1)[1].strip()

    return result


def _heuristic_faithfulness(answer: str, context: str) -> Dict[str, Any]:
    """
    Simple heuristic faithfulness check based on token overlap.
    Used as fallback when LLM judge is unavailable.
    """
    answer_tokens = set(answer.lower().split())
    context_tokens = set(context.lower().split())

    if not answer_tokens:
        return {"score": 0.0, "reasoning": "Empty answer", "verdict": "FAIL"}

    overlap = len(answer_tokens & context_tokens)
    score = overlap / len(answer_tokens)
    score = min(1.0, score)  # Cap at 1.0

    verdict = "PASS" if score >= 0.5 else "FAIL"
    reasoning = f"Token overlap: {overlap}/{len(answer_tokens)} answer tokens found in context"

    return {"score": round(score, 4), "reasoning": reasoning, "verdict": verdict}


class GenerationEvaluator:
    """
    Evaluates generation quality across a dataset of queries.
    
    Usage:
        evaluator = GenerationEvaluator()
        
        for query_data in eval_dataset:
            result = rag_pipeline.query(query_data["question"])
            evaluator.add_result(
                question=query_data["question"],
                answer=result["answer"],
                context=result["context_used"],
                expected_answer=query_data.get("expected_answer")
            )
        
        metrics = evaluator.compute_metrics()
    """

    def __init__(self, use_llm_judge: bool = True):
        """
        Args:
            use_llm_judge: Whether to use LLM-as-judge (requires Gemini API key)
        """
        self.use_llm_judge = use_llm_judge
        self.results: List[Dict[str, Any]] = []

    def add_result(self, question: str, answer: str, context: str = "",
                   expected_answer: str = None) -> Dict[str, Any]:
        """
        Evaluate a single generation result.

        Args:
            question: The user's question
            answer: The generated answer
            context: The context used for generation
            expected_answer: Optional ground-truth answer

        Returns:
            Dict of per-query metrics
        """
        generate_fn = _get_gemini_judge() if self.use_llm_judge else None

        metrics = {"question": question[:100]}

        # Faithfulness
        faith = faithfulness_score(answer, context, generate_fn)
        metrics["faithfulness_score"] = faith["score"]
        metrics["faithfulness_verdict"] = faith["verdict"]
        metrics["faithfulness_reasoning"] = faith["reasoning"]

        # Answer relevance
        relevance = answer_relevance_score(answer, question, generate_fn)
        metrics["relevance_score"] = relevance["score"]
        metrics["relevance_verdict"] = relevance["verdict"]
        metrics["relevance_reasoning"] = relevance["reasoning"]

        self.results.append(metrics)
        return metrics

    def compute_metrics(self) -> Dict[str, float]:
        """Compute averaged generation metrics across all queries."""
        if not self.results:
            return {}

        n = len(self.results)
        return {
            "num_queries": n,
            "avg_faithfulness": sum(r["faithfulness_score"] for r in self.results) / n,
            "avg_relevance": sum(r["relevance_score"] for r in self.results) / n,
            "faithfulness_pass_rate": sum(
                1 for r in self.results if r["faithfulness_verdict"] == "PASS"
            ) / n,
            "relevance_pass_rate": sum(
                1 for r in self.results if r["relevance_verdict"] == "PASS"
            ) / n,
        }

    def get_per_query_results(self) -> List[Dict[str, Any]]:
        """Get individual query results."""
        return self.results

    def reset(self):
        """Clear all stored results."""
        self.results = []

    def summary_table(self) -> str:
        """Generate a formatted summary table."""
        metrics = self.compute_metrics()
        if not metrics:
            return "No generation evaluation results."

        lines = [
            "=" * 50,
            "  GENERATION EVALUATION RESULTS",
            "=" * 50,
            f"  Queries evaluated:       {metrics['num_queries']}",
            "-" * 50,
            f"  Avg Faithfulness:        {metrics['avg_faithfulness']:.4f}",
            f"  Faithfulness Pass Rate:  {metrics['faithfulness_pass_rate']:.2%}",
            "-" * 50,
            f"  Avg Relevance:           {metrics['avg_relevance']:.4f}",
            f"  Relevance Pass Rate:     {metrics['relevance_pass_rate']:.2%}",
            "=" * 50,
        ]

        return "\n".join(lines)
