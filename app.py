# app.py
import streamlit as st
import pandas as pd
from ner_processing import ner_detection
from summarization_processing import summarize_text

st.title("NER + Summarization Dashboard")

# --- USER INPUT ---
user_input = st.text_area("Enter your text:")

# --- USER OPTIONS ---
st.subheader("Choose the tasks you want to perform:")
do_summarization = st.checkbox("Summarization", value=True)
do_ner = st.checkbox("Named Entity Recognition (NER)", value=True)

if st.button("Process Text"):
    if user_input.strip() == "":
        st.warning("Please enter some text!")
    else:
        # --- SUMMARIZATION ---
        if do_summarization:
            try:
                summary = summarize_text(user_input)
                st.subheader("Summary")
                st.write(summary)
            except Exception as e:
                st.error(f"Error in summarization: {e}")

# --- NER ---
if do_ner:
    try:
        ner_results = ner_detection(user_input)  # ✅ pass raw text, not .split("\n")

        # Convert to DataFrame for display
        df = pd.DataFrame(ner_results)

        if not df.empty:
            st.subheader("Named Entities")
            st.dataframe(df)
        else:
            st.write("No entities detected.")
    except Exception as e:
        st.error(f"Error in NER: {e}")
