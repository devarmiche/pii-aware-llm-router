"""Streamlit interface for the pipeline in src/pipeline.py — see claude.md
step 6. Thin UI layer only: all routing/anonymization/cost logic lives in
src/pipeline.py, src/router.py and src/tracker.py and is unit-tested there.
"""

import streamlit as st
from pypdf import PdfReader

from src.pipeline import API_MODEL, OLLAMA_MODEL, answer_question

st.set_page_config(page_title="pii-aware-llm-router", layout="wide")

ROUTING_MODES = {
    "Automatic": None,
    "100% Local": "local",
    "100% API": "api",
}

if "doc_text" not in st.session_state:
    st.session_state.doc_text = ""
if "doc_name" not in st.session_state:
    st.session_state.doc_name = None
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {role, content, meta}
if "session_cost_eur" not in st.session_state:
    st.session_state.session_cost_eur = 0.0


def _extract_text(uploaded_file) -> str:
    if uploaded_file.type == "application/pdf":
        reader = PdfReader(uploaded_file)
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    return uploaded_file.read().decode("utf-8")


with st.sidebar:
    st.title("pii-aware-llm-router")
    st.caption(
        "Anonymizes a document, routes the question to a local or API model "
        "under a data-sensitivity constraint, and logs the real cost."
    )
    mode_label = st.radio("Routing mode", list(ROUTING_MODES), index=0)
    force_route = ROUTING_MODES[mode_label]
    if mode_label == "Automatic":
        st.caption(
            "Any detected PII forces the local route, regardless of complexity — "
            "see src/router.py."
        )
    st.divider()
    st.metric("Session cost", f"{st.session_state.session_cost_eur:.4f} EUR")
    if st.button("Reset conversation"):
        st.session_state.messages = []
        st.session_state.session_cost_eur = 0.0
        st.rerun()

st.header("Document")
uploaded_file = st.file_uploader("Upload a PDF or text file", type=["pdf", "txt"])
if uploaded_file is not None and uploaded_file.name != st.session_state.doc_name:
    st.session_state.doc_text = _extract_text(uploaded_file)
    st.session_state.doc_name = uploaded_file.name
    st.session_state.messages = []

with st.expander(
    "Or paste document text directly", expanded=not st.session_state.doc_text
):
    pasted = st.text_area("Document text", value=st.session_state.doc_text, height=150)
    if pasted != st.session_state.doc_text:
        st.session_state.doc_text = pasted
        st.session_state.doc_name = None
        st.session_state.messages = []

if st.session_state.doc_text:
    st.caption(f"{len(st.session_state.doc_text):,} characters loaded.")

st.header("Chat")
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message["role"] == "assistant":
            meta = message["meta"]
            cols = st.columns(4)
            cols[0].metric("Route", meta["route"])
            cols[1].metric("Latency", f"{meta['latency_ms']:.0f} ms")
            cols[2].metric("Cost", f"{meta['cost_eur']:.4f} EUR")
            cols[3].metric(
                "Tokens", f"{meta['input_tokens']} in / {meta['output_tokens']} out"
            )
            with st.expander("Routing reason and anonymized text sent to the model"):
                st.write(f"**Model:** {meta['model']}")
                st.write(f"**Reason:** {meta['reason']}")
                st.text(meta["masked_doc"][:3000])

question = st.chat_input(
    "Ask a question about the document"
    if st.session_state.doc_text
    else "Upload or paste a document first"
)
if question:
    if not st.session_state.doc_text:
        st.error("Upload or paste a document before asking a question.")
    else:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.spinner(
            f"Routing to {OLLAMA_MODEL if force_route == 'local' else API_MODEL if force_route == 'api' else 'auto-selected model'}..."
        ):
            try:
                result = answer_question(
                    st.session_state.doc_text, question, force_route=force_route
                )
            except Exception as exc:
                st.session_state.messages.append(
                    {"role": "assistant", "content": f"Error: {exc}", "meta": None}
                )
                st.rerun()

        st.session_state.session_cost_eur += result.cost_eur
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result.answer,
                "meta": {
                    "route": result.decision.route,
                    "reason": result.decision.reason,
                    "model": result.model,
                    "latency_ms": result.latency_ms,
                    "cost_eur": result.cost_eur,
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                    "masked_doc": result.masked_doc,
                },
            }
        )
        st.rerun()
