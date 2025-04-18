import streamlit as st
from streamlit_option_menu import option_menu
import os
from dotenv import load_dotenv
import PyPDF2 as pdf
from docx import Document
import google.generativeai as genai
import re
from langdetect import detect
import numpy as np
import pandas as pd
from collections import Counter  # Added missing import

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Page config
st.set_page_config(page_title="AI Resume Analyzer", layout="wide")

# Supported languages
LANGUAGES = {
    "en": "English",
    "hi": "Hindi", 
    "ru": "Russian",
    "es": "Spanish",
    "fr": "French"
}

# File processors
def extract_pdf_text(file):
    reader = pdf.PdfReader(file)
    return " ".join(page.extract_text() or "" for page in reader.pages)

def extract_docx_text(file):
    return "\n".join(para.text for para in Document(file).paragraphs)

def extract_txt_text(file):
    return file.read().decode("utf-8")

def process_file(file):
    ext = file.name.split(".")[-1].lower()
    if ext == "pdf": return extract_pdf_text(file)
    elif ext == "docx": return extract_docx_text(file)
    elif ext == "txt": return extract_txt_text(file)
    else: raise ValueError(f"Unsupported file type: {ext}")

# Gemini AI Functions
def analyze_with_gemini(job_desc, resume_text, prompt, language="en"):
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    # Add language instruction if not English
    lang_instruction = f"Respond in {LANGUAGES[language]}." if language != "en" else ""
    
    response = model.generate_content(f"""
    Job Description: {job_desc}
    Resume: {resume_text}
    Task: {prompt} {lang_instruction}
    """)
    return response.text

def extract_match_percentage(text):
    match = re.search(r"(\d+)%", text)
    return match.group(1) if match else "N/A"

# Data Analysis
class ResumeInsights:
    @staticmethod
    def top_skills(text, n=15):
        words = re.findall(r'\w{4,}', text.lower())
        return pd.DataFrame(
            Counter(words).most_common(n),
            columns=['Skill', 'Frequency']
        )
    
    @staticmethod
    def score_stats(scores):
        scores = np.array(scores)
        return {
            'Average': f"{np.mean(scores):.1f}%",
            'Median': f"{np.median(scores):.1f}%",
            'Range': f"{np.min(scores)}%-{np.max(scores)}%"
        }

# Session state
if "resume" not in st.session_state:
    st.session_state.resume = ""
    st.session_state.job_desc = ""
    st.session_state.language = "en"
    st.session_state.scores = []

# Sidebar menu
with st.sidebar:
    selected = option_menu(
        "Menu",
        ["Home", "Upload", "JD", "Analyze"],
        icons=["house", "file-earmark", "file-text", "graph-up"],
        default_index=0
    )
    st.session_state.language = st.selectbox(
        "Language",
        options=list(LANGUAGES.keys()),
        format_func=lambda x: LANGUAGES[x]
    )

# Pages
if selected == "Home":
    st.title("AI Resume Analyzer")
    st.write("Upload your resume and job description to begin analysis.")

elif selected == "Upload":
    st.header("Upload Resume")
    file = st.file_uploader("Choose PDF/DOCX/TXT", type=["pdf", "docx", "txt"])
    if file:
        try:
            st.session_state.resume = process_file(file)
            st.success("Resume processed!")
        except Exception as e:
            st.error(f"Error: {str(e)}")

elif selected == "JD":
    st.header("Job Description")
    st.session_state.job_desc = st.text_area("Paste JD here")
    if st.session_state.job_desc:
        st.success("JD saved!")

elif selected == "Analyze":
    if not st.session_state.resume or not st.session_state.job_desc:
        st.warning("Upload resume and JD first")
    else:
        # Original 8 tabs with working prompts
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
            "Summary", "Match %", "Improvements", "Keywords",
            "Questions", "Suggestions", "Fit Score", "Skills"
        ])
        
        with tab1:  # Summary
            response = analyze_with_gemini(
                st.session_state.job_desc,
                st.session_state.resume,
                "Provide a detailed professional summary of this resume",
                st.session_state.language
            )
            st.write(response)
        
        with tab2:  # Match %
            match = analyze_with_gemini(
                st.session_state.job_desc,
                st.session_state.resume,
                "Compare the resume with the job description and return ONLY the match percentage (e.g., '75%')",
                st.session_state.language
            )
            match_percent = extract_match_percentage(match)
            st.session_state.scores.append(float(match_percent))
            st.metric("Match Percentage", f"{match_percent}%")
        
        with tab3:  # Improvements
            improvements = analyze_with_gemini(
                st.session_state.job_desc,
                st.session_state.resume,
                "List specific improvements needed for this resume to better match the job description",
                st.session_state.language
            )
            st.write(improvements)
        
        with tab4:  # Keywords
            keywords = analyze_with_gemini(
                st.session_state.job_desc,
                st.session_state.resume,
                "List the most important keywords missing from the resume compared to the job description",
                st.session_state.language
            )
            st.write(keywords)
        
        with tab5:  # Questions
            questions = analyze_with_gemini(
                st.session_state.job_desc,
                st.session_state.resume,
                "Generate 5 potential interview questions based on this resume and job description",
                st.session_state.language
            )
            st.write(questions)
        
        with tab6:  # Suggestions
            suggestions = analyze_with_gemini(
                st.session_state.job_desc,
                st.session_state.resume,
                "Provide 3 concrete suggestions to improve this resume for the specific job",
                st.session_state.language
            )
            st.write(suggestions)
        
        with tab7:  # Fit Score
            fit_score = analyze_with_gemini(
                st.session_state.job_desc,
                st.session_state.resume,
                "Rate the overall fit between this resume and job description on a scale of 1-10 with justification",
                st.session_state.language
            )
            st.write(fit_score)
        
        with tab8:  # Skills
            skills = analyze_with_gemini(
                st.session_state.job_desc,
                st.session_state.resume,
                "List the top 10 skills from the resume that are most relevant to the job description",
                st.session_state.language
            )
            st.write(skills)
        
        # Hidden data insights
        with st.expander("📊 Advanced Analysis"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Top Resume Skills**")
                skills_df = ResumeInsights.top_skills(st.session_state.resume)
                st.dataframe(skills_df)
            
            with col2:
                if st.session_state.scores:
                    st.write("**Match History**")
                    stats = ResumeInsights.score_stats(st.session_state.scores)
                    st.write(f"Average: {stats['Average']}")
                    st.write(f"Median: {stats['Median']}")
                    st.write(f"Range: {stats['Range']}")
                    st.line_chart(pd.DataFrame({
                        'Scores': st.session_state.scores
                    }))
      
      