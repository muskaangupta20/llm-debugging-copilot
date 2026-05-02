# Demo Results Guide

## Best Live Demo Cases
Use one of these in the Streamlit app:

1. `Python package issue`
Shows environment-aware package installation advice and is easy for anyone to understand quickly.

2. `Spark storage path issue`
Shows why retrieval matters for cloud storage path formats and permissions.

3. `Azure Synapse quota issue`
Shows the clearest difference between generic advice and grounded, platform-specific debugging help.

## Suggested Presentation Flow
1. Open `streamlit run app.py`.
2. Load a built-in example from the sidebar.
3. Point out the retrieved evidence cards and match scores.
4. Compare the RAG response against the baseline response.
5. Explain that the evaluation script measures whether RAG covers the expected debugging concepts more reliably.

## What To Highlight
- The system does not only generate an answer; it retrieves similar prior debugging cases first.
- The RAG output is grounded in structured examples with root causes, fix steps, and sources.
- The baseline output has no retrieval context, so it is more likely to stay generic.
- The app exposes retrieval confidence, which helps communicate uncertainty instead of overclaiming.

## Evaluation Artifacts
After running `python evaluation/evaluate.py`, use:
- `evaluation/results.json` for detailed machine-readable outputs
- `evaluation/report.md` for a report-ready summary

## Recommended Demo Claim
This project shows that adding retrieval and structured grounding can make debugging guidance more relevant and more actionable than direct prompting alone, especially for platform-specific student errors.
