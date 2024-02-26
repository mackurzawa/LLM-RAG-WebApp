import streamlit as st
import os

st.balloons()
st.title("LLM RAG Chat")

st.header("Firstly let me read all your additional materials. Provide them in PDF format!")

file_uploader = st.file_uploader("Upload your documents", type='pdf')

if file_uploader:
    with open(os.path.join('data', file_uploader.name), 'wb') as f:
        f.write(file_uploader.read())

st.markdown('---')
_, lets_chat_button_container, _ = st.columns([2, 1, 2])
lets_chat_button_container.button("Let's Chat!")