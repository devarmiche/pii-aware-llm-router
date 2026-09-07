import argparse
from pathlib import Path

from src.pipeline import answer_question


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask a question about a document.")
    parser.add_argument("doc", type=Path, help="path to a text document")
    parser.add_argument("question", help="question to ask about the document")
    args = parser.parse_args()

    doc_text = args.doc.read_text(encoding="utf-8")
    result = answer_question(doc_text, args.question)

    print(f"Route: {result.decision.route} ({result.decision.reason})")
    print(f"Model: {result.model}")
    print(
        f"Latency: {result.latency_ms:.0f} ms | "
        f"tokens: {result.input_tokens} in / {result.output_tokens} out | "
        f"cost: {result.cost_eur:.4f} EUR"
    )
    print()
    print("Anonymized document sent to the model:")
    print(result.masked_doc)
    print()
    print("Answer:")
    print(result.answer)


if __name__ == "__main__":
    main()
