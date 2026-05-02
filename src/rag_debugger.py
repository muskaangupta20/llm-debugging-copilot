import json
import os
from pathlib import Path
from typing import Dict, List, Optional

from openai import OpenAI

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:  # pragma: no cover - fallback for minimal environments
    TfidfVectorizer = None
    cosine_similarity = None


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "errors_dataset.json"
PROMPTS_DIR = ROOT_DIR / "prompts"
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4")
DEFAULT_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")


def get_openai_client() -> Optional[OpenAI]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


def load_dataset(path: Path = DATA_PATH) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_prompt_template(filename: str, fallback: str) -> str:
    path = PROMPTS_DIR / filename
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return fallback.strip()


def normalize_text(value: Optional[str]) -> str:
    return (value or "").strip()


def build_user_query(
    error_message: str,
    code_snippet: str = "",
    environment: str = "",
    assignment_context: str = "",
) -> str:
    parts = [
        f"Error message: {normalize_text(error_message)}",
        f"Environment: {normalize_text(environment)}",
        f"Assignment context: {normalize_text(assignment_context)}",
        f"Code snippet: {normalize_text(code_snippet)}",
    ]
    return "\n".join(part for part in parts if part.split(": ", 1)[1])


def dataset_document(item: Dict) -> str:
    fix_steps = " ".join(item.get("fix_steps", []))
    common_mistakes = " ".join(item.get("common_mistakes", []))
    tags = " ".join(item.get("tags", []))
    return "\n".join(
        [
            f"Error: {item.get('error', '')}",
            f"Platform: {item.get('platform', item.get('context', ''))}",
            f"Context: {item.get('context', '')}",
            f"Explanation: {item.get('explanation', '')}",
            f"Root cause: {item.get('root_cause', '')}",
            f"Fix steps: {fix_steps}",
            f"Common mistakes: {common_mistakes}",
            f"Tags: {tags}",
        ]
    )


def filter_candidates(dataset: List[Dict], environment: str = "") -> List[Dict]:
    environment = normalize_text(environment).lower()
    if not environment:
        return dataset

    filtered = [
        item
        for item in dataset
        if environment in item.get("platform", "").lower()
        or environment in item.get("context", "").lower()
        or environment in " ".join(item.get("tags", [])).lower()
    ]
    return filtered or dataset


def retrieve_with_tfidf(query: str, candidates: List[Dict], top_k: int) -> List[Dict]:
    documents = [dataset_document(item) for item in candidates]

    if TfidfVectorizer and cosine_similarity and documents:
        vectorizer = TfidfVectorizer(stop_words="english")
        matrix = vectorizer.fit_transform(documents + [query])
        scores = cosine_similarity(matrix[-1], matrix[:-1]).flatten()
    else:
        query_terms = set(query.lower().split())
        scores = []
        for document in documents:
            doc_terms = set(document.lower().split())
            overlap = len(query_terms & doc_terms)
            scores.append(overlap / max(len(query_terms), 1))

    ranked = sorted(
        zip(scores, candidates),
        key=lambda pair: pair[0],
        reverse=True,
    )[:top_k]

    return [{"score": float(score), "item": item} for score, item in ranked]


def retrieve_with_embeddings(
    query: str,
    candidates: List[Dict],
    top_k: int,
    client: Optional[OpenAI],
) -> List[Dict]:
    if not client:
        raise RuntimeError("OPENAI_API_KEY is required for embedding-based retrieval.")

    documents = [dataset_document(item) for item in candidates]
    embedding_input = documents + [query]
    response = client.embeddings.create(
        model=DEFAULT_EMBEDDING_MODEL,
        input=embedding_input,
    )
    vectors = [record.embedding for record in response.data]
    query_vector = [vectors[-1]]
    document_vectors = vectors[:-1]
    scores = cosine_similarity(query_vector, document_vectors).flatten()
    ranked = sorted(
        zip(scores, candidates),
        key=lambda pair: pair[0],
        reverse=True,
    )[:top_k]
    return [{"score": float(score), "item": item} for score, item in ranked]


def retrieve_similar_errors(
    query: str,
    dataset: List[Dict],
    top_k: int = 3,
    environment: str = "",
    retrieval_method: str = "tfidf",
) -> List[Dict]:
    candidates = filter_candidates(dataset, environment=environment)
    retrieval_method = retrieval_method.lower()

    if retrieval_method == "embeddings":
        if cosine_similarity is None:
            raise RuntimeError(
                "scikit-learn is required for embedding similarity. Install requirements first."
            )
        return retrieve_with_embeddings(query, candidates, top_k, get_openai_client())

    return retrieve_with_tfidf(query, candidates, top_k)


def retrieval_confidence_label(retrieved_examples: List[Dict]) -> str:
    if not retrieved_examples:
        return "low"

    top_score = retrieved_examples[0]["score"]
    if top_score >= 0.45:
        return "high"
    if top_score >= 0.2:
        return "medium"
    return "low"


def build_context(retrieved_examples: List[Dict]) -> str:
    if not retrieved_examples:
        return "No relevant examples found."

    confidence = retrieval_confidence_label(retrieved_examples)
    blocks = [f"Retrieval confidence: {confidence}"]
    for index, result in enumerate(retrieved_examples, start=1):
        item = result["item"]
        quoted_fix_steps = " | ".join(
            [f'"{step}"' for step in item.get("fix_steps", [])]
        )
        block = "\n".join(
            [
                f"Example {index}",
                f"Score: {result['score']:.3f}",
                f"Error: {item.get('error', '')}",
                f"Platform: {item.get('platform', item.get('context', ''))}",
                f"Quoted root cause: \"{item.get('root_cause', '')}\"",
                f"Explanation: {item.get('explanation', '')}",
                f"Quoted fix steps: {quoted_fix_steps}",
                f"Common mistakes: {' | '.join(item.get('common_mistakes', []))}",
                f"Source: {item.get('source_type', 'Unknown')} - {item.get('source_url', 'N/A')}",
            ]
        )
        blocks.append(block)

    return "\n\n".join(blocks)


def build_structured_result(
    error_message: str,
    code_snippet: str = "",
    environment: str = "",
    assignment_context: str = "",
) -> Dict[str, str]:
    return {
        "error_message": normalize_text(error_message),
        "code_snippet": normalize_text(code_snippet),
        "environment": normalize_text(environment),
        "assignment_context": normalize_text(assignment_context),
    }


def build_prompt_input(user_input: Dict[str, str]) -> str:
    return "\n".join(
        [
            f"Error message: {user_input['error_message'] or 'Not provided'}",
            f"Environment: {user_input['environment'] or 'Not provided'}",
            f"Assignment context: {user_input['assignment_context'] or 'Not provided'}",
            f"Code snippet:\n{user_input['code_snippet'] or 'Not provided'}",
        ]
    )


def build_baseline_prompt_input(user_input: Dict[str, str]) -> str:
    return "\n".join(
        [
            f"Error message: {user_input['error_message'] or 'Not provided'}",
            f"Code snippet:\n{user_input['code_snippet'] or 'Not provided'}",
        ]
    )


def generate_llm_response(
    user_input: Dict[str, str],
    retrieved_examples: Optional[List[Dict]] = None,
    mode: str = "rag",
    model: str = DEFAULT_MODEL,
) -> str:
    client = get_openai_client()
    if not client:
        return (
            "OPENAI_API_KEY is not set.\n"
            "Set it in your terminal first, then run again.\n"
            "Example:\n"
            "export OPENAI_API_KEY='your_api_key_here'"
        )

    baseline_template = load_prompt_template(
        "baseline_prompt.txt",
        """
        You are an Assignment Debugging Copilot for students.
        Explain technical errors clearly, identify likely root causes, and provide actionable fixes.
        If details are missing, say what should be checked rather than inventing facts.
        Formatting rules:
        - Return plain markdown only.
        - Use exactly these numbered section titles and keep them in this order.
        - Do not add extra headings before, between, or after the numbered sections.
        - Keep each section concise: 2 to 5 sentences or short bullet points.
        - Avoid long code blocks unless a command is essential.
        - Answer only from the user's provided error, environment, context, and code snippet.
        - Do not mention retrieved examples, sources, evidence, or a knowledge base.
        - If you are uncertain, say what to inspect next rather than sounding overly confident.
        Return sections titled exactly:
        1. Simple explanation
        2. Likely root cause
        3. Step-by-step fix
        4. What to check next
        5. Confidence
        """,
    )
    rag_template = load_prompt_template(
        "rag_prompt.txt",
        """
        You are an Assignment Debugging Copilot for students.
        Use the retrieved knowledge base examples as grounding context.
        Prefer the retrieved evidence when it is relevant, and mention uncertainty when the match is weak.
        Formatting rules:
        - Return plain markdown only.
        - Use exactly these numbered section titles and keep them in this order.
        - Do not add extra headings before, between, or after the numbered sections.
        - Keep each section concise: 2 to 5 sentences or short bullet points.
        - Avoid long code blocks unless a command is essential.
        - Base your answer on the retrieved examples whenever they are relevant.
        - In the fix section, explicitly tie important recommendations to Example 1, Example 2, or Example 3 when possible.
        - Reuse the retrieved quoted root-cause and quoted fix-step fields explicitly when they are relevant.
        - If the evidence is weak, say that clearly and avoid pretending the retrieval strongly supports the answer.
        - In "Similar past error from the knowledge base", mention whether the retrieved match is strong, medium, or weak.
        - In "Sources used", list only the retrieved source names or URLs that are actually relevant.
        - Do not answer as if this were a baseline prompt; the point is to use and reference retrieved evidence.
        Return sections titled exactly:
        1. Simple explanation
        2. Likely root cause
        3. Step-by-step fix
        4. What to check next
        5. Similar past error from the knowledge base
        6. Sources used
        7. Confidence
        """,
    )

    prompt_parts = [build_baseline_prompt_input(user_input)]
    system_prompt = baseline_template

    if mode == "rag":
        system_prompt = rag_template
        prompt_parts = [build_prompt_input(user_input)]
        prompt_parts.append(
            "Retrieved examples from the debugging knowledge base:\n"
            f"{build_context(retrieved_examples or [])}"
        )
        prompt_parts.append(
            "Important: use the retrieved examples as evidence. "
            "When a recommendation is supported by a retrieved case, name the supporting example number explicitly."
        )
    else:
        prompt_parts.append(
            "Important: answer from direct reasoning only using the user's provided inputs. "
            "Do not mention sources, retrieved examples, or any knowledge base."
        )

    user_prompt = "\n\n".join(prompt_parts)

    try:
        response = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.output_text
    except Exception as exc:  # pragma: no cover - external API surface
        return f"Error while calling OpenAI API: {exc}"


def print_retrieval_results(retrieved_examples: List[Dict]) -> None:
    print("\nTop Retrieved Matches:")
    if not retrieved_examples:
        print("No retrieved matches found.")
        return

    for index, result in enumerate(retrieved_examples, start=1):
        item = result["item"]
        print(
            f"{index}. [{result['score']:.3f}] "
            f"{item.get('error', '')} "
            f"({item.get('platform', item.get('context', ''))})"
        )


def collect_multiline_input(prompt_text: str) -> str:
    print(prompt_text)
    print("Press Enter on an empty line to finish.")
    lines = []
    while True:
        line = input()
        if not line.strip():
            break
        lines.append(line)
    return "\n".join(lines)


def run_debugger(
    error_message: str,
    environment: str = "",
    assignment_context: str = "",
    code_snippet: str = "",
    top_k: int = 3,
    retrieval_method: str = "tfidf",
    include_baseline: bool = True,
) -> Dict:
    dataset = load_dataset()
    user_input = build_structured_result(
        error_message=error_message,
        code_snippet=code_snippet,
        environment=environment,
        assignment_context=assignment_context,
    )
    query = build_user_query(**user_input)
    retrieved_examples = retrieve_similar_errors(
        query=query,
        dataset=dataset,
        top_k=top_k,
        environment=environment,
        retrieval_method=retrieval_method,
    )

    result = {
        "user_input": user_input,
        "retrieval_method": retrieval_method,
        "retrieval_confidence": retrieval_confidence_label(retrieved_examples),
        "retrieved_examples": retrieved_examples,
        "rag_response": generate_llm_response(
            user_input,
            retrieved_examples=retrieved_examples,
            mode="rag",
        ),
    }

    if include_baseline:
        result["baseline_response"] = generate_llm_response(
            user_input,
            mode="baseline",
        )

    return result


def main() -> None:
    print("LLM Assignment Debugging Copilot")
    print("-" * 40)
    error_message = input("Paste the main error message:\n> ").strip()
    environment = input("Environment/platform (optional):\n> ").strip()
    assignment_context = input("Assignment context (optional):\n> ").strip()
    retrieval_method = (
        input("Retrieval method [tfidf/embeddings] (default tfidf):\n> ").strip().lower()
        or "tfidf"
    )
    code_snippet = collect_multiline_input("Paste a short code snippet if helpful (optional):")

    result = run_debugger(
        error_message=error_message,
        environment=environment,
        assignment_context=assignment_context,
        code_snippet=code_snippet,
        retrieval_method=retrieval_method,
    )

    print_retrieval_results(result["retrieved_examples"])
    print(f"\nRetrieval confidence: {result['retrieval_confidence']}")

    print("\n" + "=" * 60)
    print("RAG Response")
    print("-" * 60)
    print(result["rag_response"])

    if "baseline_response" in result:
        print("\n" + "=" * 60)
        print("Baseline Response")
        print("-" * 60)
        print(result["baseline_response"])

    print("=" * 60)


if __name__ == "__main__":
    main()
