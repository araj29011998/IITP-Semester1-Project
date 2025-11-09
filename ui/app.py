import os, time, requests, streamlit as st
API_URL = os.getenv("API_URL","http://api:8000")

st.set_page_config(page_title="TNF Agent", layout="wide")
st.title("TNF Agent (Local)")
tabs = st.tabs(["Assistant","Debug"])

with tabs[0]:
    q = st.text_input("Ask about TNF docs, SOPs, KPIs, etc.")
    if st.button("Go") and q:
        t0 = time.time()
        r = requests.post(f"{API_URL}/ask", json={"query": q})
        dt = time.time()-t0
        if r.ok:
            data = r.json()
            st.markdown("### Answer")
            st.write(data["answer"])
            st.markdown("### Sources")
            for s in data.get("sources", []):
                st.write(f"- **{s.get('file','?')}** page {s.get('page','?')}")
            st.caption(f"Latency: {dt:.2f}s")
        else:
            st.error(f"Error: {r.status_code} {r.text}")

with tabs[1]:
    st.subheader("Debug")
    try:
        hr = requests.get(f"{API_URL}/health", timeout=5).json()
        st.write("API health:", hr)
    except Exception as e:
        st.error(f"API not reachable: {e}")
    st.write("Models:", os.getenv("CHAT_MODEL","llama3.1:8b"),
             "| Embeddings:", os.getenv("EMBED_MODEL","nomic-embed-text"))
