import streamlit as st
from notebooks.task3_modular import load_vector_store, generate_answer

# Page Config
st.set_page_config(page_title="CrediTrust AI Insights", layout="wide")

st.title("🛡️ CrediTrust Complaint Analyzer")
st.markdown("Analyze customer feedback using Local AI (Llama 3.2)")

# 1. Load the data (Cached so it's fast)
@st.cache_resource
def init_system():
    return load_vector_store()

index, metadata = init_system()

# 2. Sidebar for Settings
st.sidebar.header("Settings")
k_value = st.sidebar.slider("Number of Complaints to Analyze", 1, 10, 5)

# 3. User Input
query = st.text_input("Ask a question about customer complaints:", 
                     placeholder="e.g., Why are people unhappy with Credit Cards?")

if query:
    with st.spinner("Analyzing complaints..."):
        # 1. Get the data from your Task 3 function
        answer, sources = generate_answer(query, index, metadata)
        
        # --- NEW: VISUAL SUMMARY (Added at the top of results) ---
        st.subheader("📊 Executive Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("Primary Concern", "Debt/Credit")
        col2.metric("Evidence Samples", len(sources))
        col3.metric("Sentiment", "Critical")
        st.divider()

        # 2. Show the Detailed Analysis
        st.subheader("📝 Detailed Analysis")
        st.markdown(answer)

        # --- NEW: DOWNLOAD BUTTON (Added below the analysis) ---
        st.download_button(
            label="📩 Download Full Report as TXT",
            data=answer,
            file_name="creditrust_analysis_report.txt",
            mime="text/plain"
        )

    # 3. Keep the Raw Evidence at the bottom
    with st.expander("🔍 View Raw Evidence (Found in Database)"):
        for i, doc in enumerate(sources):
            st.info(f"**Source {i+1} (ID: {doc.get('complaint_id')})**")
            st.write(doc.get('text'))
            st.divider()