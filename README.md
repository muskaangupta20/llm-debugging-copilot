# LLM Assignment Debugging Copilot

## 📌 Overview
This project builds an **LLM-based debugging assistant** designed to help students working on programming and data-related assignments (Python, Spark, Azure, etc.) resolve errors more efficiently.

Students often struggle with unclear error messages, misconfigurations, and scattered documentation. This tool aims to **bridge that gap** by providing:
- Simple explanations of errors  
- Root cause analysis  
- Step-by-step debugging guidance  

## 🎯 Problem Statement
Debugging is one of the most time-consuming and frustrating parts of learning programming and data engineering. Error messages are often cryptic, and solutions are scattered across documentation and forums.

This project addresses this problem by building an intelligent assistant that helps students understand and fix errors quickly, improving both learning outcomes and productivity.

## 💡 Solution
The system takes an **error message and context** as input and generates:
1. Clear explanation (in simple terms)
2. Root cause of the issue
3. Step-by-step fix
4. Common mistakes to avoid

## ⚙️ Methodology
This project uses **Retrieval-Augmented Generation (RAG)** combined with **prompt engineering**.

### Why RAG?
- Debugging requires external knowledge (docs, past errors)
- RAG retrieves relevant information before generating responses
- Produces more accurate and context-aware answers than generic LLM outputs

## 🏗️ System Architecture (High-Level)

User Input (Error + Code)
↓
Retriever (FAISS / ChromaDB)
↓
Relevant Error-Solution Context
↓
LLM (Prompt Engineered)
↓
Structured Output (Explanation + Fix)

## 📊 Data Sources
- Python, Spark, Azure documentation  
- Stack Overflow discussions  
- Curated dataset of real debugging errors (from coursework)  

> Note: No personal or sensitive data is used.

## 🛠️ Tech Stack
- Python  
- OpenAI API / Open-source LLM (Mistral/LLaMA)  
- FAISS / ChromaDB (Vector Database)  
- LangChain (optional)  

## 📂 Project Structure

llm-debugging-copilot/
├── data/ # Error datasets
├── src/ # Core code
├── notebooks/ # Experiments
├── prompts/ # Prompt templates
├── evaluation/ # Evaluation scripts/results
├── README.md
└── requirements.txt


## 📈 Evaluation
The system will be evaluated against a baseline of **direct LLM prompting (no retrieval)**.

### Metrics:
- Correctness of suggested fix  
- Relevance to the error  
- Usefulness (actionable steps)  

A successful system should outperform the baseline by providing **more accurate and context-aware debugging guidance**.

## 🚀 Future Work
- Expand dataset with more real-world errors  
- Add multi-language support  
- Integrate with IDE (VS Code extension)  
- Improve reasoning with agent-based workflows  

## 👤 Author
Muskaan Gupta
MS Data Science, Boston University  

## ⭐ Why This Project Matters
This project combines **LLMs + real-world debugging challenges**, making it highly practical and directly useful for students and developers.

It goes beyond generic AI tools by focusing on **context-aware, explainable debugging assistance**.