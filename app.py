import os
import json
import time
import streamlit as st
from google import genai
from google.genai import types
from google.genai.errors import APIError
from pydantic import BaseModel, Field
from typing import List, Optional
import dotenv 
from pypdf import PdfReader # NEW: Import PdfReader for PDF processing

# --- 1. Define the Structured Output Schema using Pydantic ---
class Feedback(BaseModel):
    """Detailed feedback sections for resume improvement."""
    keywordMatch: str = Field(description="Actionable feedback on missing and used keywords from the Job Description and overall relevance.")
    contentImpact: str = Field(description="Actionable feedback on utilizing strong action verbs, quantifying results, and overall professional impact.")
    formattingAndStructure: str = Field(description="Actionable feedback on resume parsability, standard section headings, and layout/structure issues that might confuse an ATS.")

class ATSResult(BaseModel):
    """The final structured ATS analysis report."""
    score: int = Field(description="The ATS compatibility score from 0 to 100, where higher is better.")
    summary: str = Field(description="A brief, encouraging summary of the analysis results.")
    feedback: Feedback = Field(description="Structured, detailed feedback for improvement.")

# --- Helper Function for PDF Extraction ---

def extract_text_from_pdf(uploaded_file):
    """Extracts text content from a PDF file uploaded to Streamlit."""
    try:
        reader = PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            # Attempt to extract text from each page
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
            else:
                # If extraction fails (e.g., image-only PDF), log a warning
                st.warning("⚠️ Warning: Could not extract text from one or more pages. Ensure your PDF is text-selectable, not an image scan.")
        if not text.strip():
            st.error("❌ Failed to extract any readable text from the PDF. Is it an image-only file?")
            return None
        return text
    except Exception as e:
        st.error(f"❌ An error occurred during PDF reading: {e}")
        return None

# --- 2. Configuration and Core Logic (Adapted for Streamlit) ---

@st.cache_resource(show_spinner=False)
def get_gemini_client():
    """Initializes and returns the Gemini Client, checking for the API key."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("❌ Gemini API Key not found!")
        st.info("Please set the GEMINI_API_KEY environment variable (or ensure it's loaded from your .env file) to run the analysis.")
        return None
    
    try:
        client = genai.Client()
        return client
    except Exception as e:
        st.error(f"❌ Failed to initialize Gemini Client: {e}")
        return None

def analyze_resume_ats(client, resume_text: str, jd_text: str):
    """
    Connects to the Gemini API to analyze a resume against a job description
    and returns a structured ATS report.
    """
    
    # Define the core instructions for the model
    system_prompt = (
        "You are a world-class Applicant Tracking System (ATS) analyst with 20 years of "
        "experience. Your task is to evaluate a candidate's resume against a specific job "
        "description. Analyze the documents across three core areas: Keyword Match, "
        "Content Impact (quantifiable achievements, action verbs), and Formatting/Structure "
        "(ATS parsability, standard headings). Generate a compatibility score from 0 to 100 "
        "and provide detailed, actionable feedback. Ensure all feedback is professional and "
        "constructive. The output MUST adhere strictly to the provided JSON schema."
    )
    
    user_query = f"""
    Analyze the following RESUME against the JOB DESCRIPTION.

    ---
    RESUME TEXT:
    {resume_text}
    ---
    JOB DESCRIPTION:
    {jd_text}
    """
    
    contents = [
        types.Content(role="user", parts=[types.Part(text=user_query)])
    ]
    
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_json_schema=ATSResult.model_json_schema(),
        system_instruction=system_prompt, 
        temperature=0.2,
    )
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash-preview-09-2025',
                contents=contents, 
                config=config,
            )

            json_data = response.text
            ats_report = ATSResult.model_validate_json(json_data)
            return ats_report

        except APIError as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                time.sleep(wait_time)
            else:
                st.error(f"Failed to get a response after {max_retries} attempts. API Error: {e}")
                return None
        except (json.JSONDecodeError, ValueError) as e:
            st.error(f"Failed to parse AI response (Invalid JSON structure). Error: {e}")
            return None
        except Exception as e:
            st.error(f"An unexpected error occurred during analysis: {e}")
            return None
    return None

# --- 3. Streamlit UI Components ---

def get_score_color_style(score):
    """Returns CSS color based on score for visualization."""
    if score >= 80:
        return "green"
    elif score >= 60:
        return "orange"
    return "red"

def display_report(report: ATSResult):
    """Displays the ATS report using Streamlit columns and markdown."""
    
    st.markdown("---")
    st.subheader("✅ ATS Compatibility Report")
    
    # Score Card
    col1, col2 = st.columns([1, 4])
    with col1:
        st.markdown(f"""
        <div style="
            text-align: center; 
            padding: 15px; 
            border-radius: 10px; 
            border: 3px solid {get_score_color_style(report.score)};
            background-color: {get_score_color_style(report.score)}1A;
        ">
            <h1 style="font-size: 4em; color: {get_score_color_style(report.score)}; margin: 0;">{report.score}%</h1>
            <p style="font-size: 1.2em; margin: 0; color: #333333;">Match Score</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.success(f"**Summary:** {report.summary}")
    
    st.markdown("---")
    
    # Detailed Feedback Tabs
    tab1, tab2, tab3 = st.tabs(["⭐ Keyword Match", "💪 Content Impact", "📐 Formatting"])

    with tab1:
        st.markdown("**Focus on integrating specific keywords from the Job Description.**")
        st.info(report.feedback.keywordMatch)

    with tab2:
        st.markdown("**Ensure every bullet point highlights quantifiable achievements.**")
        st.info(report.feedback.contentImpact)

    with tab3:
        st.markdown("**Maintain standard section headers and avoid complex visual elements.**")
        st.info(report.feedback.formattingAndStructure)

def main_app():
    """Main Streamlit application function."""
    
    dotenv.load_dotenv()

    st.set_page_config(page_title="ATS Resume Analyzer", layout="wide")
    
    st.title("🤖 AI-Powered ATS Resume Scorer")
    

    client = get_gemini_client()
    
    if not client:
        return # Stop execution if client initialization fails (missing API key)

    # Input Areas
    col_resume, col_jd = st.columns(2)
    
    with col_resume:
        # File uploader for the resume
        uploaded_file = st.file_uploader(
            "📋 Upload Your Resume (PDF only)",
            type=['pdf'],
            help="Upload your resume file. Text will be extracted for analysis."
        )

        # Resume text initialization
        resume_text = None
        if uploaded_file is not None:
            # Extract text from the uploaded PDF
            with st.spinner("Extracting text from PDF..."):
                resume_text = extract_text_from_pdf(uploaded_file)
            
            if resume_text:
                st.info("✅ Text extracted successfully. You can review the extracted text below.")
                with st.expander("Extracted Resume Text Preview"):
                    st.text(resume_text)


    with col_jd:
        # Job Description remains a text area
        jd_text = st.text_area(
            "💼 Paste the Target Job Description Here",
            height=400,
            value="", # FIX: Changed default value to an empty string
            help="The AI will compare your resume against these required skills and responsibilities. Please paste the full text of the job description."
        )

    st.markdown("---")

    if st.button("🚀 Analyze Resume & Get ATS Score", use_container_width=True, type="primary"):
        if uploaded_file is None:
            st.error("Please upload a PDF file for your resume.")
        elif not resume_text:
            st.error("Could not proceed: Failed to extract readable text from the uploaded PDF.")
        elif not jd_text:
            st.error("Please enter the Job Description text.")
        else:
            with st.spinner("Analyzing resume... This may take a moment as the AI evaluates keywords and structure."):
                report = analyze_resume_ats(client, resume_text, jd_text)
                
                if report:
                    display_report(report)

if __name__ == "__main__":
    main_app()