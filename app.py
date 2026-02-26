import streamlit as st
import os
import json
from extractor import extract_text_from_pdf_bytes
from processor import process_chargesheet_text

def main():
    st.set_page_config(page_title="Smart Chargesheet Assistant", page_icon="⚖️", layout="wide")
    
    st.title("⚖️ Smart Chargesheet Review & Summarisation Assistant")
    st.markdown("Upload a Hindi Case Diary (Police Chargesheet) to generate a structured case summary, classify the crime, and verify missing procedural documents.")
    
    # Sidebar for API Key
    with st.sidebar:
        st.header("Configuration")
        api_key = st.text_input("Google Gemini API Key", type="password", help="Get your API key from Google AI Studio")
        if not api_key:
            api_key = os.environ.get("GEMINI_API_KEY", "")
            
        st.markdown("---")
        st.markdown("### About")
        st.markdown("Built for the AI/ML Hackathon Stage 1 Requirements.")
        st.markdown("Powered by **Gemini 1.5 Flash**")
        
    uploaded_file = st.file_uploader("Upload Case Diary PDF", type=["pdf"])
    
    if uploaded_file is not None:
        if st.button("Analyze Chargesheet"):
            if not api_key:
                st.error("Please provide a Google Gemini API Key in the sidebar or via the GEMINI_API_KEY environment variable.")
                return
                
            with st.spinner("Extracting text from PDF (handling OCR/Hindi text)..."):
                raw_text = extract_text_from_pdf_bytes(uploaded_file.read())
                
            if not raw_text:
                st.error("Could not extract text from the provided PDF.")
                return
                
            with st.spinner("Analyzing text with Gemini AI Model..."):
                try:
                    result = process_chargesheet_text(raw_text, api_key)
                except Exception as e:
                    st.error(f"Error processing text: {e}")
                    return
            
            if not result:
                st.error("Failed to generate a valid response from the AI model.")
                return
                
            # Display Outputs
            st.success("Analysis Complete!")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.header("Output A — Structured Case Summary")
                summary = result.get("summary", {})
                st.markdown(f"**FIR Number & Date:** {summary.get('fir_number', 'N/A')}")
                st.markdown(f"**Police Station:** {summary.get('police_station', 'N/A')}")
                st.markdown(f"**Accused:** {', '.join(summary.get('accused_names', ['N/A']))}")
                st.markdown(f"**Victim/Complainant:** {', '.join(summary.get('victim_names', ['N/A']))}")
                st.markdown(f"**Legal Sections Applied:** {', '.join(summary.get('legal_sections', ['N/A']))}")
                
                with st.expander("Comprehensive Incident Facts", expanded=True):
                    st.write(summary.get('incident_facts', 'N/A'))
                    
            with col2:
                st.header("Output B — Crime Classification")
                classification = result.get("classification", {})
                c_type = classification.get("crime_type", "UNKNOWN")
                
                # Colorize based on match
                if c_type == "UNKNOWN":
                    st.warning(f"**Classification:** {c_type}")
                else:
                    st.info(f"**Classification:** {c_type}")
                    
                st.markdown(f"**Reason:** {classification.get('reason', 'N/A')}")
                
            st.markdown("---")
            st.header("Output C — Missing Items Checklist")
            st.markdown("Compared extracted content against the static JSON schema for the detected crime type.")
            
            checklist = result.get("checklist", [])
            if not checklist:
                st.info("No checklist items generated or crime type was UNKNOWN.")
            else:
                for item in checklist:
                    status = item.get("status", "").upper()
                    name = item.get("item_name", "Unknown Item")
                    details = item.get("details", "")
                    
                    if status == "PRESENT":
                        st.success(f"✅ **{name}**: {details}")
                    elif status == "PARTIAL":
                        st.warning(f"⚠️ **{name}**: {details}")
                    else:
                        st.error(f"❌ **{name}**: {details if details else 'Not detected'}")
                        
            with st.expander("View Raw JSON Output"):
                st.json(result)

if __name__ == "__main__":
    main()
