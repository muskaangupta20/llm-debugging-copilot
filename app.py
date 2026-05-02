import html
import json
import re
from pathlib import Path

import streamlit as st

from src.rag_debugger import run_debugger


EXAMPLES = {
    "Azure Storage permission mismatch": {
        "error_message": "AuthorizationPermissionMismatch: This request is not authorized to perform this operation using this permission",
        "environment": "Azure Storage",
        "assignment_context": "Writing assignment output files to ADLS Gen2 using a service principal.",
        "code_snippet": "service_client.get_file_system_client('assignment-data').get_file_client('outputs/result.csv').upload_data(data)",
    },
    "Azure Synapse quota issue": {
        "error_message": "InvalidHttpRequestToLivy: Your Spark job requested 12 vcores but workspace has 0 core limit",
        "environment": "Azure Synapse",
        "assignment_context": "Running an assignment notebook in a student Synapse workspace.",
        "code_snippet": "spark.conf.set('spark.executor.cores', '4')\nspark.conf.set('spark.executor.instances', '3')",
    },
    "Spark storage path issue": {
        "error_message": "AnalysisException: Path does not exist",
        "environment": "Spark",
        "assignment_context": "Reading parquet files from ADLS in a Spark notebook.",
        "code_snippet": "df = spark.read.parquet('abfss://data@account.dfs.core.windows.net/missing-folder/')",
    },
    "Azure Data Factory 404": {
        "error_message": "HttpRequestFailedWithClientError: 404 NotFound",
        "environment": "Azure Data Factory",
        "assignment_context": "Calling a REST API from a Data Factory pipeline for a class assignment.",
        "code_snippet": "response = pipeline.run(endpoint='/students/assignment/results')",
    },
    "Python package issue": {
        "error_message": "ModuleNotFoundError: No module named 'pandas'",
        "environment": "Python",
        "assignment_context": "Working on a data analysis homework in a virtual environment.",
        "code_snippet": "import pandas as pd\ndf = pd.read_csv('sales.csv')",
    },
}


SECTION_PATTERN = re.compile(r"^\s*(?:#+\s*)?(\d+)\.\s+(.+?)\s*$")
RESULTS_PATH = Path(__file__).resolve().parent / "evaluation" / "results.json"


def apply_style() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(204, 230, 255, 0.55), transparent 35%),
                radial-gradient(circle at top right, rgba(255, 226, 179, 0.45), transparent 30%),
                linear-gradient(180deg, #f7f4ee 0%, #eef3f7 100%);
        }
        .hero {
            padding: 1.4rem 1.6rem;
            border-radius: 18px;
            background: linear-gradient(135deg, rgba(15, 76, 117, 0.92), rgba(26, 95, 122, 0.9));
            color: white;
            border: 1px solid rgba(255, 255, 255, 0.18);
            box-shadow: 0 18px 40px rgba(12, 48, 64, 0.16);
            margin-bottom: 1rem;
        }
        .hero h1 {
            margin: 0;
            font-size: 2rem;
            letter-spacing: -0.03em;
        }
        .hero p {
            margin: 0.55rem 0 0 0;
            font-size: 1rem;
            opacity: 0.92;
        }
        .note-card {
            padding: 1rem 1.1rem;
            border-radius: 16px;
            background: rgba(255, 255, 255, 0.72);
            border: 1px solid rgba(15, 76, 117, 0.1);
            box-shadow: 0 10px 24px rgba(30, 41, 59, 0.08);
            margin-bottom: 1rem;
        }
        .section-label {
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #36627a;
            margin-bottom: 0.35rem;
            font-weight: 700;
        }
        .result-chip {
            display: inline-block;
            padding: 0.3rem 0.65rem;
            border-radius: 999px;
            background: #dceef8;
            color: #114a63;
            font-weight: 600;
            margin-right: 0.4rem;
            margin-bottom: 0.4rem;
        }
        .response-card {
            padding: 1rem 1.1rem;
            border-radius: 18px;
            background: rgba(255, 255, 255, 0.78);
            border: 1px solid rgba(15, 76, 117, 0.12);
            box-shadow: 0 10px 24px rgba(30, 41, 59, 0.08);
            margin-bottom: 0.9rem;
        }
        .response-card h4 {
            margin: 0 0 0.45rem 0;
            font-size: 1rem;
            line-height: 1.3;
            color: #16384f;
        }
        .response-card p {
            margin: 0;
            line-height: 1.65;
        }
        .response-label {
            font-size: 0.76rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #537287;
            font-weight: 700;
            margin-bottom: 0.35rem;
        }
        .stMarkdown pre, .stCodeBlock pre {
            white-space: pre-wrap !important;
            word-break: break-word !important;
            overflow-wrap: anywhere !important;
        }
        .stMarkdown code {
            white-space: pre-wrap !important;
            word-break: break-word !important;
            overflow-wrap: anywhere !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def initialize_session() -> None:
    defaults = {
        "error_message": "",
        "environment": "",
        "assignment_context": "",
        "code_snippet": "",
        "result": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def load_example(name: str) -> None:
    example = EXAMPLES[name]
    st.session_state.error_message = example["error_message"]
    st.session_state.environment = example["environment"]
    st.session_state.assignment_context = example["assignment_context"]
    st.session_state.code_snippet = example["code_snippet"]


def load_evaluation_summary() -> dict | None:
    if not RESULTS_PATH.exists():
        return None
    try:
        payload = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
        return payload.get("summary")
    except Exception:
        return None


def render_retrieved_examples(result: dict) -> None:
    st.subheader("Retrieved Evidence")
    for index, example in enumerate(result["retrieved_examples"], start=1):
        item = example["item"]
        with st.container(border=True):
            st.markdown(
                f"**{index}. {item.get('error', '')}**  \n"
                f"`score={example['score']:.3f}`  `platform={item.get('platform', '')}`"
            )
            st.write(item.get("explanation", ""))
            st.write(f"Root cause: {item.get('root_cause', '')}")
            st.write("Fix steps:")
            for step in item.get("fix_steps", []):
                st.write(f"- {step}")
            st.caption(f"Source: {item.get('source_type', 'unknown')} | {item.get('source_url', 'N/A')}")


def render_why_this_answer(result: dict) -> None:
    st.subheader("Why This Answer?")
    if not result["retrieved_examples"]:
        st.info("No retrieved evidence was available, so the answer is relying on general model reasoning.")
        return

    top_example = result["retrieved_examples"][0]
    item = top_example["item"]
    st.markdown(
        f"""
        <div class="note-card">
          <div class="section-label">Primary Evidence</div>
          <p><strong>Top retrieved example:</strong> {html.escape(item.get('error', 'Unknown'))}</p>
          <p><strong>Match score:</strong> {top_example['score']:.3f} | <strong>Confidence:</strong> {html.escape(result['retrieval_confidence'].title())}</p>
          <p><strong>Retrieved root cause:</strong> {html.escape(item.get('root_cause', ''))}</p>
          <p><strong>Retrieved fix signal:</strong> {html.escape(item.get('fix_steps', [''])[0] if item.get('fix_steps') else 'No fix step available.')}</p>
          <p><strong>Source:</strong> {html.escape(item.get('source_url', 'N/A'))}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_evaluation_summary() -> None:
    summary = load_evaluation_summary()
    st.subheader("Evaluation Snapshot")
    if not summary:
        st.caption("")
        return

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Test cases", summary["num_cases"])
    col2.metric("Avg RAG score", f"{summary['average_rag_score']:.3f}")
    col3.metric("Avg baseline score", f"{summary['average_baseline_score']:.3f}")
    col4.metric("Avg delta", f"{summary['average_score_delta']:.3f}")

    st.caption(
        f"RAG scored higher on {summary['rag_better_cases']} cases, tied on {summary['tied_cases']}, "
        f"and scored lower on {summary['baseline_better_cases']}."
    )


def parse_response_sections(response_text: str) -> list[tuple[str, str]]:
    if not response_text.strip():
        return []

    sections = []
    current_title = "Response"
    current_lines = []

    for line in response_text.splitlines():
        match = SECTION_PATTERN.match(line)
        if match:
            if current_lines:
                sections.append((current_title, "\n".join(current_lines).strip()))
            current_title = match.group(2).strip()
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_title, "\n".join(current_lines).strip()))

    return [(title, body) for title, body in sections if title or body]


def render_response_blocks(title: str, response_text: str, raw_key: str) -> None:
    st.markdown(f"#### {title}")
    sections = parse_response_sections(response_text)

    if not sections:
        st.markdown(response_text)
    else:
        for index, (section_title, body) in enumerate(sections, start=1):
            safe_title = html.escape(section_title)
            safe_body = html.escape(body).replace("\n", "<br>")
            st.markdown(
                f"""
                <div class="response-card">
                  <div class="response-label">Section {index}</div>
                  <h4>{safe_title}</h4>
                  <p>{safe_body}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with st.expander(f"Raw {title}", expanded=False):
        st.code(response_text, language="markdown")


st.set_page_config(
    page_title="LLM Debugging Copilot",
    page_icon="🛠️",
    layout="wide",
)

apply_style()
initialize_session()

st.markdown(
    """
    <div class="hero">
      <h1>LLM Assignment Debugging Copilot</h1>
      <p>Context-aware debugging support for Python, Spark, and Azure assignments with retrieved evidence, root-cause analysis, and step-by-step fixes.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Run Settings")
    retrieval_method = st.selectbox(
        "Retrieval method",
        options=["tfidf", "embeddings"],
        index=0,
        help="Embeddings uses the OpenAI embeddings API and requires OPENAI_API_KEY.",
    )
    top_k = st.slider("Top retrieved examples", min_value=1, max_value=5, value=3)
    include_baseline = st.checkbox(
        "Show baseline comparison (evaluation only)",
        value=False,
        help="Keep this off for the main RAG demo. Turn it on only when you want to compare against direct prompting.",
    )
    st.divider()
    st.subheader("Quick Demo Examples")
    example_name = st.selectbox("Load example", ["Custom input"] + list(EXAMPLES.keys()))
    if st.button("Apply example", use_container_width=True) and example_name != "Custom input":
        load_example(example_name)
    if st.button("Clear inputs", use_container_width=True):
        load_example("Azure Storage permission mismatch")
        st.session_state.error_message = ""
        st.session_state.environment = ""
        st.session_state.assignment_context = ""
        st.session_state.code_snippet = ""
        st.session_state.result = None

intro_col, input_col = st.columns([0.8, 1.2], gap="large")

with intro_col:
    st.markdown(
        """
        <div class="note-card">
          <div class="section-label">Primary System</div>
          <p>This app is designed to showcase the RAG debugging copilot itself: retrieved evidence, grounded reasoning, source-aware fixes, and match confidence.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="note-card">
          <div class="section-label">What This App Does</div>
          <p>This assistant takes a debugging error, retrieves similar cases from a curated knowledge base, and generates grounded guidance with explanations, likely causes, and next steps.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="note-card">
          <div class="section-label">How The Assistant Works</div>
          <p>It combines your error message, code snippet, and assignment context with retrieved debugging examples. The model then uses that evidence to produce a structured RAG response and confidence-aware guidance.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_evaluation_summary()

with input_col:
    st.subheader("Student Input")
    error_message = st.text_area(
        "Error message",
        key="error_message",
        height=140,
        placeholder="Paste the main traceback or platform error here.",
    )
    row1, row2 = st.columns(2)
    with row1:
        environment = st.text_input(
            "Environment / platform",
            key="environment",
            placeholder="Python, Spark, Azure Synapse...",
        )
    with row2:
        assignment_context = st.text_input(
            "Assignment context",
            key="assignment_context",
            placeholder="Brief context about the failing task",
        )
    code_snippet = st.text_area(
        "Code snippet (optional)",
        key="code_snippet",
        height=220,
        placeholder="Paste the failing code block here.",
    )
    run_button = st.button("Analyze Error", type="primary", use_container_width=True)


if run_button:
    if not st.session_state.error_message.strip():
        st.error("Add an error message before running the analysis.")
    else:
        with st.spinner("Running retrieval and generating debugging guidance..."):
            st.session_state.result = run_debugger(
                error_message=st.session_state.error_message,
                environment=st.session_state.environment,
                assignment_context=st.session_state.assignment_context,
                code_snippet=st.session_state.code_snippet,
                top_k=top_k,
                retrieval_method=retrieval_method,
                include_baseline=include_baseline,
            )


if st.session_state.result:
    result = st.session_state.result
    st.success("Analysis complete.")

    if result["retrieval_confidence"] == "low":
        st.warning(
            "Low-confidence retrieval: the retrieved matches are weak, so the response may rely more on general model reasoning than domain-specific evidence."
        )
    elif result["retrieval_confidence"] == "medium":
        st.info(
            "Medium-confidence retrieval: the answer uses relevant evidence, but the match is not highly specific."
        )

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    metric_col1.metric("Retrieval method", result["retrieval_method"])
    metric_col2.metric("Confidence", result["retrieval_confidence"].title())
    metric_col3.metric("Retrieved examples", len(result["retrieved_examples"]))
    top_score = result["retrieved_examples"][0]["score"] if result["retrieved_examples"] else 0.0
    metric_col4.metric("Top match score", f"{top_score:.3f}")

    st.markdown(
        " ".join(
            [
                f"<span class='result-chip'>{result['retrieval_method'].upper()}</span>",
                f"<span class='result-chip'>{result['retrieval_confidence'].title()} confidence</span>",
                f"<span class='result-chip'>Top-{len(result['retrieved_examples'])} evidence</span>",
            ]
        ),
        unsafe_allow_html=True,
    )

    render_why_this_answer(result)
    render_retrieved_examples(result)

    st.subheader("RAG Debugging Guidance")
    if "baseline_response" in result:
        rag_tab, baseline_tab = st.tabs(["RAG Response", "Baseline Comparison"])
        with rag_tab:
            render_response_blocks("RAG Response", result["rag_response"], "rag_raw")
        with baseline_tab:
            render_response_blocks("Baseline Response", result["baseline_response"], "baseline_raw")
    else:
        render_response_blocks("RAG Response", result["rag_response"], "rag_raw")

    with st.expander("Raw run output"):
        st.code(json.dumps(result, indent=2), language="json")
