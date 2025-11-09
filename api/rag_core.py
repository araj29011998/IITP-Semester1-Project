import os, httpx
from vectordb import search_chunks
from tools import TOOL_REGISTRY  # <-- tools

LLM_BASE = os.getenv("LLM_BASE","http://ollama:11434")
CHAT_MODEL = os.getenv("CHAT_MODEL","llama3.1:8b")

SYSTEM = (
"You are a helpful TNF assistant. Prefer accurate, concise answers. "
"Always cite as [file:page]. Never fabricate."
)

def ollama_chat(messages, model=CHAT_MODEL):
    payload = {"model": model, "messages": messages, "stream": False}
    with httpx.Client(timeout=120) as cx:
        r = cx.post(f"{LLM_BASE}/v1/chat/completions", json=payload)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

def format_ctx(results):
    blocks = []
    for text, meta in results:
        src = f"{meta.get('file','?')}:{meta.get('page','?')}"
        blocks.append(f"[{src}] {text}")
    return "\n\n".join(blocks) if blocks else "(no context)"

def ask(query: str, session_id: str | None = None):
    # ---------- SQL fast-path: run tool only, skip LLM ----------
    if "sql" in TOOL_REGISTRY and TOOL_REGISTRY["sql"].should_run(query):
        tool_out = TOOL_REGISTRY["sql"].run(query)
        return {"answer": tool_out, "sources": [], "tool_output": tool_out}

    # ---------- Normal RAG ----------
    ctx = search_chunks(query, k=int(os.getenv("MAX_CHUNKS","6")))
    ctx_text = format_ctx(ctx)
    messages = [
        {"role":"system","content": SYSTEM},
        {"role":"user","content": f"Context:\n{ctx_text}\n\nQuestion: {query}\nAnswer with citations."}
    ]
    try:
        answer = ollama_chat(messages)
    except httpx.HTTPError as e:
        answer = f"(LLM error: {e}; showing context)\n\n{ctx_text}"
    sources = [c[1] for c in ctx]
    return {"answer": answer, "sources": sources}
