import json
import os
from pathlib import Path
from difflib import SequenceMatcher
from openai import OpenAI


DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "errors_dataset.json"


def load_dataset(path=DATA_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def retrieve_similar_error(user_error: str, dataset: list, top_k: int = 3):
    scored = []

    for item in dataset:
        score = similarity(user_error, item["error"])
        scored.append((score, item))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]


def build_context(retrieved_examples: list) -> str:
    if not retrieved_examples:
        return "No relevant examples found."

    blocks = []
    for i, (_, item) in enumerate(retrieved_examples, start=1):
        block = f"""
Example {i}
Error: {item['error']}
Context: {item['context']}
Explanation: {item['explanation']}
Solution: {item['solution']}
""".strip()
        blocks.append(block)

    return "\n\n".join(blocks)


def generate_llm_response(user_error: str, retrieved_examples: list) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return (
            "OPENAI_API_KEY is not set.\n"
            "Set it in your terminal first, then run again.\n"
            "Example:\n"
            "export OPENAI_API_KEY='your_api_key_here'"
        )

    client = OpenAI(api_key=api_key)
    context = build_context(retrieved_examples)

    system_prompt = (
        "You are an Assignment Debugging Copilot for students. "
        "Your job is to explain technical errors clearly and give practical fixes. "
        "Use the retrieved examples as grounding context. "
        "Do not invent environment details you do not know. "
        "If uncertain, say what needs to be checked."
    )

    user_prompt = f"""
A student encountered this error:

{user_error}

Here are similar retrieved examples from the debugging knowledge base:

{context}

Please answer in this exact structure:

1. Simple explanation
2. Likely root cause
3. Step-by-step fix
4. What to check next
5. Similar past error from the knowledge base
""".strip()

    try:
        response = client.responses.create(
            model="gpt-5.4",
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.output_text

    except Exception as e:
        return f"Error while calling OpenAI API: {e}"


def main():
    dataset = load_dataset()

    print("LLM Assignment Debugging Copilot")
    print("-" * 40)
    user_error = input("Paste your error message here:\n> ")

    results = retrieve_similar_error(user_error, dataset, top_k=3)

    print("\nTop Retrieved Matches:")
    for i, (score, item) in enumerate(results, start=1):
        print(f"{i}. [{score:.2f}] {item['error']} ({item['context']})")

    print("\n" + "=" * 60)
    llm_answer = generate_llm_response(user_error, results)
    print(llm_answer)
    print("=" * 60)


if __name__ == "__main__":
    main()