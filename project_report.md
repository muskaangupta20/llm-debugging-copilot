# LLM Assignment Debugging Copilot  
### Muskaan Gupta  
Boston University  
MS Data Science  

## 1. Introduction

Debugging is one of the most time-consuming and frustrating aspects of learning programming and data engineering. Students working with technologies such as Python, Apache Spark, and Azure often encounter unclear error messages, configuration issues, and scattered documentation. These challenges slow down learning and reduce productivity.

This project proposes an **LLM-based Assignment Debugging Copilot**, designed to help students interpret error messages and receive structured debugging guidance. Unlike generic AI tools, this system uses **Retrieval-Augmented Generation (RAG)** to ground responses in relevant prior debugging cases, improving both accuracy and usefulness.

The goal of this project is to help students:
- Understand errors in simple terms  
- Identify root causes  
- Follow step-by-step fixes  
- Learn from similar debugging scenarios  


## 2. Approach

### 2.1 System Overview

The system takes structured input from the user:
- Error message  
- Environment/platform (Python, Spark, Azure)  
- Assignment context  
- Optional code snippet  

The system retrieves relevant debugging cases and uses an LLM to generate structured guidance.

### 2.2 Architecture
User Input
↓
Retriever (TF-IDF / Embeddings)
↓
Top-k Similar Errors
↓
LLM (Baseline or RAG Mode)
↓
Structured Debugging Output


### 2.3 Retrieval Methods

Two retrieval strategies were implemented:

**TF-IDF similarity**
- Fast and lightweight  
- Works well for exact matches  

**Embedding-based retrieval**
- Uses OpenAI embeddings  
- Captures semantic similarity  
- Performs better on paraphrased errors  

### 2.4 Generation Methods

Two approaches were compared:

**Baseline prompting**
- Direct LLM response without retrieval  
- Often generic and less accurate  

**RAG prompting**
- Uses retrieved examples as context  
- Produces grounded and structured outputs  

### 2.5 Dataset Design

A structured debugging dataset was created with the following fields:
- error  
- platform  
- context  
- explanation  
- root_cause  
- fix_steps  
- common_mistakes  

The dataset includes examples from:
- Python errors  
- Spark errors  
- Azure Synapse and Data Factory issues  

This structured format improves both retrieval quality and response grounding.

## 3. Evaluation

### 3.1 Evaluation Setup

Evaluation is performed using a test set of unseen debugging cases located in:

`evaluation/test_cases.json`

The system compares:
- Baseline LLM responses  
- RAG-based responses  

### 3.2 Metrics

The following qualitative metrics are used:

- **Correctness** – Accuracy of the suggested fix  
- **Relevance** – Alignment with the input error  
- **Usefulness** – Clarity and actionability  

### 3.3 Results

| Metric       | Baseline | RAG |
|-------------|---------|-----|
| Correctness | 6/10    | 9/10 |
| Relevance   | 7/10    | 9/10 |
| Usefulness  | 6/10    | 9/10 |

### 3.4 Key Findings

- RAG significantly improves answer quality  
- Responses are more specific and less generic  
- Retrieval reduces hallucinations  
- Outputs are more structured and actionable  

### 3.5 Error Analysis

Failure cases include:
- Unseen errors not present in the dataset  
- Weak retrieval matches  
- Ambiguous input lacking sufficient context  

## 4. Results and Discussion

The system demonstrates that combining retrieval with LLM generation improves debugging assistance. Compared to baseline prompting, the RAG system produces clearer explanations, better reasoning, and more actionable debugging steps.

Embedding-based retrieval further improves performance by capturing semantic similarity between errors. The addition of a Streamlit interface enhances usability and enables interactive demonstrations.

## 5. Ethics and Limitations

- The system may generate incorrect debugging suggestions (hallucinations)  
- It should not replace expert debugging in critical systems  
- Limited dataset reduces coverage of rare errors  
- Performance depends heavily on retrieval quality  

## 6. Future Work

- Expand dataset with real-world debugging logs  
- Add citation-based responses from official documentation  
- Integrate as a VS Code extension  
- Improve evaluation using automated scoring  
- Support more programming languages  

## 7. References

- OpenAI API Documentation  
- Python Documentation  
- Apache Spark Documentation  
- Microsoft Azure Documentation  

## 8. Acknowledgements / Contribution

This project was completed individually by Muskaan Gupta.

All components of the project, including dataset creation, system design, implementation, evaluation, and report writing, were completed independently.