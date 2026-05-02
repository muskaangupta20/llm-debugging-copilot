import json
from pathlib import Path
from statistics import mean
from typing import Dict, List

from src.rag_debugger import run_debugger


ROOT_DIR = Path(__file__).resolve().parent.parent
TEST_CASES_PATH = ROOT_DIR / "evaluation" / "test_cases.json"
RESULTS_PATH = ROOT_DIR / "evaluation" / "results.json"
REPORT_PATH = ROOT_DIR / "evaluation" / "report.md"


def load_test_cases(path: Path = TEST_CASES_PATH) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def keyword_score(text: str, expected_focus: List[str]) -> float:
    text_lower = text.lower()
    hits = sum(1 for keyword in expected_focus if keyword.lower() in text_lower)
    return hits / max(len(expected_focus), 1)


def retrieval_hit_score(retrieved_examples: List[Dict], expected_focus: List[str]) -> float:
    combined = " ".join(
        [
            example["item"].get("error", "")
            + " "
            + example["item"].get("root_cause", "")
            + " "
            + " ".join(example["item"].get("fix_steps", []))
            for example in retrieved_examples
        ]
    ).lower()
    hits = sum(1 for keyword in expected_focus if keyword.lower() in combined)
    return hits / max(len(expected_focus), 1)


def evaluate_case(case: Dict) -> Dict:
    result = run_debugger(
        error_message=case["error_message"],
        environment=case.get("environment", ""),
        assignment_context=case.get("assignment_context", ""),
        code_snippet=case.get("code_snippet", ""),
        top_k=3,
        retrieval_method=case.get("retrieval_method", "tfidf"),
        include_baseline=True,
    )
    expected_focus = case.get("expected_focus", [])
    rag_score = keyword_score(result["rag_response"], expected_focus)
    baseline_score = keyword_score(result["baseline_response"], expected_focus)
    retrieval_score = retrieval_hit_score(result["retrieved_examples"], expected_focus)

    return {
        "id": case["id"],
        "error_message": case["error_message"],
        "environment": case.get("environment", ""),
        "expected_focus": expected_focus,
        "retrieval_method": result["retrieval_method"],
        "retrieval_confidence": result["retrieval_confidence"],
        "retrieval_score": retrieval_score,
        "rag_score": rag_score,
        "baseline_score": baseline_score,
        "score_delta": rag_score - baseline_score,
        "top_retrievals": [
            {
                "error": example["item"].get("error", ""),
                "platform": example["item"].get("platform", ""),
                "score": round(example["score"], 3),
            }
            for example in result["retrieved_examples"]
        ],
        "rag_response": result["rag_response"],
        "baseline_response": result["baseline_response"],
    }


def summarize_results(results: List[Dict]) -> Dict:
    return {
        "num_cases": len(results),
        "average_retrieval_score": round(mean(item["retrieval_score"] for item in results), 3),
        "average_rag_score": round(mean(item["rag_score"] for item in results), 3),
        "average_baseline_score": round(mean(item["baseline_score"] for item in results), 3),
        "average_score_delta": round(mean(item["score_delta"] for item in results), 3),
        "rag_better_cases": sum(1 for item in results if item["score_delta"] > 0),
        "tied_cases": sum(1 for item in results if item["score_delta"] == 0),
        "baseline_better_cases": sum(1 for item in results if item["score_delta"] < 0),
    }


def build_report(summary: Dict, case_results: List[Dict]) -> str:
    lines = [
        "# Evaluation Report",
        "",
        "## Summary",
        f"- Number of cases: {summary['num_cases']}",
        f"- Average retrieval score: {summary['average_retrieval_score']:.3f}",
        f"- Average RAG score: {summary['average_rag_score']:.3f}",
        f"- Average baseline score: {summary['average_baseline_score']:.3f}",
        f"- Average score delta (RAG - baseline): {summary['average_score_delta']:.3f}",
        f"- Cases where RAG scored higher: {summary['rag_better_cases']}",
        f"- Cases tied: {summary['tied_cases']}",
        f"- Cases where baseline scored higher: {summary['baseline_better_cases']}",
        "",
        "## Case Breakdown",
    ]

    for case in case_results:
        lines.extend(
            [
                "",
                f"### {case['id']}",
                f"- Environment: {case['environment']}",
                f"- Error: `{case['error_message']}`",
                f"- Expected focus: {', '.join(case['expected_focus'])}",
                f"- Retrieval confidence: {case['retrieval_confidence']}",
                f"- Retrieval score: {case['retrieval_score']:.2f}",
                f"- RAG score: {case['rag_score']:.2f}",
                f"- Baseline score: {case['baseline_score']:.2f}",
                f"- Score delta: {case['score_delta']:.2f}",
                "- Top retrievals:",
            ]
        )
        for item in case["top_retrievals"]:
            lines.append(
                f"  - [{item['score']:.3f}] {item['error']} ({item['platform']})"
            )

    return "\n".join(lines) + "\n"


def main() -> None:
    test_cases = load_test_cases()
    case_results = [evaluate_case(case) for case in test_cases]
    summary = summarize_results(case_results)
    payload = {"summary": summary, "cases": case_results}

    RESULTS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    REPORT_PATH.write_text(build_report(summary, case_results), encoding="utf-8")

    print("Baseline vs RAG Evaluation")
    print("=" * 60)
    print(json.dumps(summary, indent=2))

    for case in case_results:
        print("\nCase:", case["id"])
        print("-" * 60)
        print("Expected focus:", ", ".join(case["expected_focus"]))
        print("Retrieval confidence:", case["retrieval_confidence"])
        print("Retrieval score:", f"{case['retrieval_score']:.2f}")
        print("RAG score:", f"{case['rag_score']:.2f}")
        print("Baseline score:", f"{case['baseline_score']:.2f}")
        print("Score delta:", f"{case['score_delta']:.2f}")
        print("Top retrievals:")
        for index, item in enumerate(case["top_retrievals"], start=1):
            print(f"  {index}. [{item['score']:.3f}] {item['error']} ({item['platform']})")

    print("\nSaved detailed results to", RESULTS_PATH)
    print("Saved markdown report to", REPORT_PATH)


if __name__ == "__main__":
    main()
