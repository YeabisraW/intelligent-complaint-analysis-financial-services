import streamlit as st
from notebooks.task3_modular import load_vector_store, generate_answer

# Page Config
st.set_page_config(page_title="CrediTrust AI Insights", layout="wide")

# 1. Sidebar - Reset Button and Settings
st.sidebar.header("Controls")
if st.sidebar.button("🔄 Reset Analysis"):
    st.rerun()

st.sidebar.divider()
k_value = st.sidebar.slider("Number of Complaints to Analyze", 1, 10, 5)

st.title("🛡️ CrediTrust Complaint Analyzer")
st.markdown("Analyze customer feedback using Local AI (Llama 3.2)")

# 2. Load the data (Cached so it's fast)
@st.cache_resource
def init_system():
    return load_vector_store()

index, metadata = init_system()

# 3. User Input Form (Reviewer Request: Explicit Submit Button)
with st.form("query_form"):
    query = st.text_input(
        "Ask a question about customer complaints:", 
        placeholder="e.g., Why are people unhappy with Credit Cards?"
    )
    submit_button = st.form_submit_button("📊 Run Analysis")

# 4. Logic Execution
if submit_button and query:
    with st.spinner("Analyzing complaints..."):
        # Get the data from Task 3
        answer, sources = generate_answer(query, index, metadata)
        
        # --- VISUAL SUMMARY ---
        st.subheader("📊 Executive Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("Topic", "Financial Services")
        col2.metric("Evidence Samples", len(sources))
        col3.metric("Status", "Analysis Complete")
        st.divider()

        # --- DETAILED ANALYSIS ---
        st.subheader("📝 Detailed Analysis")
        st.markdown(answer)

        # --- DOWNLOAD BUTTON ---
        st.download_button(
            label="📩 Download Full Report as TXT",
            data=answer,
            file_name="creditrust_analysis_report.txt",
            mime="text/plain"
        )

    # --- RAW EVIDENCE ---
    with st.expander("🔍 View Raw Evidence (Found in Database)"):
        for i, doc in enumerate(sources):
            st.info(f"**Source {i+1} (ID: {doc.get('complaint_id')})**")
            st.write(doc.get('text'))
            st.divider()