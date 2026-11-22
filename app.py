"""Step 7: chat UI showing answer + cited passages. Run: streamlit run app.py"""
import streamlit as st
from rag import answer

st.set_page_config(page_title="sklearn Docs Assistant", page_icon="📚")
st.title("📚 scikit-learn Docs Assistant")
st.caption("Answers only from the indexed scikit-learn User Guide, with citations.")

if "history" not in st.session_state:
    st.session_state.history = []

for turn in st.session_state.history:
    with st.chat_message("user"):
        st.write(turn["q"])
    with st.chat_message("assistant"):
        st.write(turn["r"]["answer"])
        for s in turn["r"]["sources"]:
            with st.expander(f"[{s['n']}] {s['title']} › {s['section']}"):
                st.write(s["text"])
                st.markdown(f"[Open source page]({s['url']})")

if q := st.chat_input("Ask about scikit-learn..."):
    with st.spinner("Searching docs..."):
        r = answer(q)
    st.session_state.history.append({"q": q, "r": r})
    st.rerun()
