import streamlit as st
import pandas as pd
import pdfplumber
import requests
import numpy as np
import time
from typing import List, Dict, Any, Tuple

st.title("📊 Simple Financial Document Reader")

# Upload PDF
pdf_file = st.file_uploader("Upload a PDF", type=["pdf"])
if pdf_file is not None:
    with pdfplumber.open(pdf_file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() + "\n"
        st.subheader("Extracted PDF Text")
        st.write(text)

# Upload Excel
excel_file = st.file_uploader("Upload an Excel File", type=["xlsx"])
if excel_file is not None:
    df = pd.read_excel(excel_file)
    st.subheader("Excel Data")
    st.dataframe(df)

import streamlit as st
import pandas as pd
import pdfplumber
import re
import json
from io import BytesIO
import requests
import numpy as np
from typing import List, Dict, Any, Tuple

# ---------- helper functions ----------
def parse_amount(s):
    """Turn a string like '$1,234.56' or '(1,234)' into a float (handle parentheses => negative)."""
    if s is None:
        return None
    s = str(s)
    neg = False
    if '(' in s and ')' in s:
        neg = True
    # Remove everything except digits, dot and minus
    cleaned = re.sub(r'[^0-9.\-]', '', s)
    if cleaned == '' or cleaned == '.' or cleaned == '-':
        return None
    try:
        val = float(cleaned)
    except:
        return None
    if neg:
        val = -val
    return val

def search_in_text(text, keywords):
    """Find a numeric amount near common keyword occurrences in text."""
    if not text:
        return None
    for kw in keywords:
        # pattern finds "Keyword: $ 1,234" or "Keyword 1,234"
        regex = rf'{kw}\s*(?:[:\-]|\s)?\s*([\$\(\)\d,.\s]+)'
        m = re.search(regex, text, flags=re.IGNORECASE)
        if m:
            val = parse_amount(m.group(1))
            if val is not None:
                return val
    # fallback: check lines containing the keyword
    for line in text.splitlines():
        for kw in keywords:
            if kw.lower() in line.lower():
                m = re.search(r'[\$\(\)\d,.\s]+', line)
                if m:
                    val = parse_amount(m.group(0))
                    if val is not None:
                        return val
    return None

def extract_from_pdf(uploaded_file):
    """Extract text and tables from PDF, then search for revenue/expenses/profit."""
    data = {}
    all_text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            ptext = page.extract_text()
            if ptext:
                all_text += ptext + "\n"
            # extract simple tables and append their text
            try:
                tables = page.extract_tables()
            except:
                tables = []
            for table in tables:
                # convert to dataframe when possible and append text
                try:
                    df = pd.DataFrame(table[1:], columns=table[0])
                except Exception:
                    df = pd.DataFrame(table)
                all_text += " " + df.astype(str).apply(lambda r: " ".join(r.values), axis=1).str.cat(sep=" ") + "\n"

    data['Revenue']  = search_in_text(all_text, ['revenue','total revenue','sales','turnover','net revenue'])
    data['Expenses'] = search_in_text(all_text, ['expense','expenses','total expenses','cost of sales','costs'])
    data['Profit']   = search_in_text(all_text, ['profit','net income','net profit','operating profit','earnings'])
    return data

def extract_from_excel(uploaded_file):
    """Read all sheets, try to find columns or values that match keywords."""
    data = {}
    try:
        sheets = pd.read_excel(uploaded_file, sheet_name=None)
    except Exception as e:
        st.error(f"Could not read Excel file: {e}")
        return {"Revenue": None, "Expenses": None, "Profit": None}

    combined_text = ""
    # look for explicit columns named like 'Revenue', 'Expenses', 'Profit' across sheets
    for sheet_name, df in sheets.items():
        # append a textual representation for fallback search
        combined_text += df.astype(str).apply(lambda r: " ".join(r.values), axis=1).str.cat(sep=" ") + " "
        # look for column name matches
        for col in df.columns:
            col_low = str(col).lower()
            # Revenue-like columns
            if any(k in col_low for k in ['revenue','sales','turnover','net revenue']):
                # try to sum numeric values or take first numeric cell
                nums = pd.to_numeric(df[col], errors='coerce').dropna()
                if not nums.empty:
                    data.setdefault('Revenue', float(nums.sum()) if len(nums) > 1 else float(nums.iloc[0]))
            if any(k in col_low for k in ['expense','expenses','cost','operating expenses']):
                nums = pd.to_numeric(df[col], errors='coerce').dropna()
                if not nums.empty:
                    data.setdefault('Expenses', float(nums.sum()) if len(nums) > 1 else float(nums.iloc[0]))
            if any(k in col_low for k in ['profit','net income','earnings','net profit']):
                nums = pd.to_numeric(df[col], errors='coerce').dropna()
                if not nums.empty:
                    data.setdefault('Profit', float(nums.sum()) if len(nums) > 1 else float(nums.iloc[0]))

    # fallback: parse combined text
    data.setdefault('Revenue',  search_in_text(combined_text, ['revenue','total revenue','sales','turnover','net revenue']))
    data.setdefault('Expenses', search_in_text(combined_text, ['expense','expenses','total expenses','costs','cost of sales']))
    data.setdefault('Profit',   search_in_text(combined_text, ['profit','net income','earnings','net profit']))
    return data

# ---------- Ollama + RAG utilities ----------
def is_ollama_running(base_url: str) -> bool:
    try:
        r = requests.get(base_url + "/api/tags", timeout=2)
        return r.status_code == 200
    except Exception:
        return False

def ollama_embed(base_url: str, model: str, input_texts: List[str]) -> np.ndarray:
    payload = {"model": model, "input": input_texts}
    r = requests.post(base_url + "/api/embeddings", json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    # Ollama returns {embeddings: [[...], [...]]} or {embedding: [...]} depending on client
    if "embeddings" in data:
        vectors = data["embeddings"]
    elif "embedding" in data:
        vectors = [data["embedding"]]
    else:
        raise RuntimeError("Unexpected embeddings response from Ollama")
    return np.array(vectors, dtype=np.float32)

def ollama_chat(base_url: str, model: str, system_prompt: str, messages: List[Dict[str, str]]) -> str:
    body = {
        "model": model,
        "messages": ([{"role": "system", "content": system_prompt}] + messages),
        "stream": False,
        "options": {"temperature": 0.2}
    }
    r = requests.post(base_url + "/api/chat", json=body, timeout=180)
    r.raise_for_status()
    data = r.json()
    return data.get("message", {}).get("content", "")

def chunk_text(text: str, max_chars: int = 1200, overlap: int = 200) -> List[str]:
    if not text:
        return []
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + max_chars)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == n:
            break
        start = end - overlap
        if start < 0:
            start = 0
    return [c.strip() for c in chunks if c.strip()]

def build_index(texts: List[str], embed_fn) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    if not texts:
        return np.zeros((0, 1), dtype=np.float32), []
    embs = embed_fn(texts)
    # normalize for cosine similarity
    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8
    embs = embs / norms
    metas = [{"text": t} for t in texts]
    return embs, metas

def search_index(query: str, embed_query_fn, matrix: np.ndarray, metas: List[Dict[str, Any]], k: int = 5) -> List[Dict[str, Any]]:
    if matrix is None or matrix.shape[0] == 0:
        return []
    q = embed_query_fn([query])[0]
    q = q / (np.linalg.norm(q) + 1e-8)
    sims = matrix @ q
    idxs = np.argsort(-sims)[:k]
    results = []
    for idx in idxs:
        item = {**metas[int(idx)], "score": float(sims[int(idx)])}
        results.append(item)
    return results

# ---------- Streamlit UI ----------
st.title("📄 Financial Document Extractor")

# Sidebar settings
with st.sidebar:
    st.header("Settings")
    ollama_base = st.text_input("Ollama base URL", value="http://localhost:11434")
    llm_model = st.text_input("Chat model", value="llama3.2:3b")
    embed_model = st.text_input("Embedding model", value="nomic-embed-text")
    top_k = st.slider("Top-K chunks", min_value=2, max_value=10, value=5)
    st.caption("Ollama must be running locally. Use small models for speed.")

uploaded = st.file_uploader("Upload a PDF or Excel (.xlsx) file", type=["pdf","xlsx"])
if uploaded is not None:
    ext = uploaded.name.split('.')[-1].lower()
    if ext == "pdf":
        with st.spinner("Extracting PDF..."):
            extracted = extract_from_pdf(uploaded)
    else:  # xlsx
        with st.spinner("Reading Excel..."):
            extracted = extract_from_excel(uploaded)

    st.subheader("Auto-extracted values")
    st.json(extracted)

    st.subheader("Confirm / Edit extracted values")
    # default to 0.0 if None so inputs don't break
    rev_val  = extracted.get("Revenue")
    exp_val  = extracted.get("Expenses")
    prof_val = extracted.get("Profit")

    rev = st.number_input("Revenue", value=float(rev_val) if rev_val is not None else 0.0, format="%.2f")
    exp = st.number_input("Expenses", value=float(exp_val) if exp_val is not None else 0.0, format="%.2f")
    prof = st.number_input("Profit", value=float(prof_val) if prof_val is not None else 0.0, format="%.2f")

    final = {"Revenue": rev, "Expenses": exp, "Profit": prof}
    st.write("Final structured data:")
    st.json(final)

    # allow download
    st.download_button("Download JSON", data=json.dumps(final, indent=2), file_name="financial_data.json", mime="application/json")

    # -------- Build text corpus for RAG --------
    try:
        uploaded.seek(0)
    except Exception:
        pass

    doc_text = ""
    try:
        if ext == "pdf":
            with pdfplumber.open(uploaded) as pdf:
                for page in pdf.pages:
                    t = page.extract_text() or ""
                    doc_text += t + "\n"
        else:
            sheets = pd.read_excel(uploaded, sheet_name=None)
            for _, df in sheets.items():
                doc_text += df.astype(str).apply(lambda r: " ".join(r.values), axis=1).str.cat(sep=" ") + "\n"
    except Exception as e:
        st.warning(f"Could not reconstruct full text from file for retrieval: {e}")

    # Combine with structured values for grounding
    kv_text = "\n".join([f"Revenue: {final['Revenue']}", f"Expenses: {final['Expenses']}", f"Profit: {final['Profit']}"])
    corpus = (doc_text or "") + "\n" + kv_text

    # -------- Build / cache vector index in session --------
    if "rag_index" not in st.session_state:
        st.session_state["rag_index"] = None
        st.session_state["rag_meta"] = None

    if is_ollama_running(ollama_base):
        chunks = chunk_text(corpus)
        if chunks:
            with st.spinner("Embedding and indexing document..."):
                try:
                    embed_fn = lambda texts: ollama_embed(ollama_base, embed_model, texts)
                    matrix, metas = build_index(chunks, embed_fn)
                    st.session_state["rag_index"] = matrix
                    st.session_state["rag_meta"] = metas
                    st.success(f"Indexed {len(chunks)} chunks for retrieval.")
                except Exception as e:
                    st.error(f"Embedding/indexing failed: {e}")
    else:
        st.warning("Ollama is not running or unreachable. Start it with 'ollama serve'.")

    # -------- Chat UI --------
    st.subheader("Ask questions about your document")
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    for m in st.session_state["messages"]:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    user_q = st.chat_input("Ask about revenue, expenses, profit, trends...")
    if user_q:
        st.session_state["messages"].append({"role": "user", "content": user_q})
        with st.chat_message("user"):
            st.markdown(user_q)

        answer_text = ""
        if is_ollama_running(ollama_base) and st.session_state.get("rag_index") is not None:
            try:
                embed_query_fn = lambda texts: ollama_embed(ollama_base, embed_model, texts)
                results = search_index(user_q, embed_query_fn, st.session_state["rag_index"], st.session_state["rag_meta"], k=top_k)
                context = "\n\n".join([f"[Chunk {i+1} | score={r['score']:.3f}]\n{r['text']}" for i, r in enumerate(results)])
                system = (
                    "You are a helpful financial analysis assistant. Answer strictly using the provided context. "
                    "If the answer is not present, say you cannot find it in the document. Keep answers concise."
                )
                messages = st.session_state["messages"].copy()
                messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_q}"})
                with st.spinner("Thinking..."):
                    answer_text = ollama_chat(ollama_base, llm_model, system, messages)
            except Exception as e:
                answer_text = f"Error during QA: {e}"
        else:
            answer_text = "Retrieval not available (Ollama down or index missing)."

        st.session_state["messages"].append({"role": "assistant", "content": answer_text})
        with st.chat_message("assistant"):
            st.markdown(answer_text)
