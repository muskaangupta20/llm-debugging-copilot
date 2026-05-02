# LLM Assignment Debugging Copilot

## Overview
This project builds an LLM-based debugging assistant for students working on programming and data assignments in Python, Spark, Azure Synapse, and Azure Data Factory environments.

The primary system is a retrieval-augmented debugging copilot. It retrieves similar prior debugging cases and uses them to ground the response with root causes, fix steps, and source-aware guidance. A baseline direct-prompting mode is included for evaluation, but the main product experience is the RAG assistant itself.

## Project Goal
Students often get stuck on:
- Python import, type, and file path errors
- Spark path and storage configuration issues
- Azure quota, permission, and API request problems

The goal is to help students recover faster by providing:
- a simple explanation of the error
- a likely root cause
- step-by-step debugging actions
- source-grounded guidance from similar past cases

## Implemented Features
The current system includes:
- a reusable debugging pipeline in `src/rag_debugger.py`
- structured student inputs:
  - error message
  - environment/platform
  - assignment context
  - optional code snippet
- retrieval over a curated debugging knowledge base
- two retrieval modes:
  - TF-IDF for local experiments
  - OpenAI embeddings for stronger semantic matching
- two answer modes:
  - baseline prompting without retrieval
  - RAG prompting with retrieved evidence and sources
- retrieval confidence labels
- a Streamlit demo UI for presentations
- a "Why This Answer?" evidence panel in the app
- weak/medium retrieval confidence warnings in the app
- an in-app evaluation snapshot when evaluation results are available
- an optional baseline mode for evaluation
- an evaluation script that scores baseline vs RAG
- a markdown evaluation report for writeups

## Architecture
```text
Student Input
  ├── Error message
  ├── Environment
  ├── Assignment context
  └── Code snippet
        ↓
Retriever
  ├── TF-IDF similarity
  └── Optional embeddings retrieval
        ↓
Top-k debugging cases
        ↓
LLM
  ├── Baseline mode
  └── RAG mode
        ↓
Structured debugging guidance
```

## Repository Structure
```text
llm-debugging-copilot/
├── app.py
├── data/
│   └── errors_dataset.json
├── evaluation/
│   ├── evaluate.py
│   ├── report.md
│   ├── results.json
│   └── test_cases.json
├── prompts/
│   ├── baseline_prompt.txt
│   └── rag_prompt.txt
├── src/
│   └── rag_debugger.py
├── demo_results.md
├── README.md
└── requirements.txt
```

## Dataset Design
Each debugging example contains:
- `error`
- `platform`
- `context`
- `tags`
- `explanation`
- `root_cause`
- `fix_steps`
- `common_mistakes`
- `source_type`
- `source_url`

This structure supports better retrieval and more grounded responses than a simple error-message dictionary.

## How To Run
### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set your API key
```bash
export OPENAI_API_KEY="your_api_key_here"
```

Optional model overrides:
```bash
export OPENAI_MODEL="gpt-5.4"
export OPENAI_EMBEDDING_MODEL="text-embedding-3-small"
```

### 3. Run the CLI debugger
```bash
python src/rag_debugger.py
```

### 4. Run the demo app
```bash
streamlit run app.py
```

The app highlights:
- retrieved evidence and source-aware grounding
- retrieval confidence and weak-match warnings
- a "Why This Answer?" panel showing the top retrieved evidence
- optional baseline comparison for evaluation only

### 5. Run the evaluation
```bash
python evaluation/evaluate.py
```

This generates:
- `evaluation/results.json`
- `evaluation/report.md`

## Demo Tips
For the strongest presentation, use the built-in Streamlit examples:
- Python package issue
- Spark storage path issue
- Azure Synapse quota issue

Use the app primarily as a RAG demo:
- show the retrieved evidence
- point to the "Why This Answer?" panel
- explain the grounded RAG response
- mention retrieval confidence if the match is medium or low
- mention that baseline comparison is available as an optional evaluation view

## Evaluation Approach
The evaluation compares:
- baseline prompting without retrieval
- retrieval-augmented prompting with grounded examples

The script measures:
- retrieval score against expected debugging concepts
- baseline answer coverage
- RAG answer coverage
- score delta between RAG and baseline

This is still a lightweight evaluation, but it makes the project more rigorous and easier to discuss in a final report.

## Why This Is Stronger Than The Initial Prototype
Compared with the original version, this project now has:
- richer structured input
- a reusable retrieval and generation pipeline
- optional semantic retrieval
- a polished demo interface
- a stronger evaluation story
- report-ready artifacts for submission

## Best Next Extensions
If you want to push it further after this version:
- expand the dataset with more real errors from coursework
- add citation snippets from official docs
- use an LLM judge rubric for deeper evaluation
- run a small user study comparing baseline vs RAG usefulness
- package the system as a VS Code or notebook assistant

## Author
Muskaan Gupta  
MS Data Science, Boston University
