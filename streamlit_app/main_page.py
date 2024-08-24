import streamlit as st

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
    layout="wide",
)

# Set the title of the main page
st.write("# Welcome to Zihao Wang's Dashboard! 👋")

st.sidebar.success("Select a page.")

st.markdown(
    """
    This is the main page for the data analytics app of BlueCrest Capital Project.
    """
)
