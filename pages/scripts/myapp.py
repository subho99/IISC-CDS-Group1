import streamlit as st
from pathlib import Path

st.markdown(
    """
    <style>
    .reportview-container {
        background: url("mauro-lima-w5QDlbjJwEY-unsplash.jpg");
    }
   </style>
    """,
    unsafe_allow_html=True
)

def test_function():
    st.write("I'm declared in a different package but succesffuly inovked by this")
    st.write(Path.cwd())
    return Path.cwd()