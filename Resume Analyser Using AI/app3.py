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
from sklearn.feature_extraction.text import CountVectorizer

# Load env variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Streamlit page config
st.set_page_config(page_title="AI Resume Analyzer", layout="wide")

# Supported languages
SUPPORTED_LANGUAGES = {
    "en": "English", "hi": "Hindi", "ru": "Russian",
    "es": "Spanish", "fr": "French", "de": "German"
}

# ----- File Processing -----
def extract_text_from_pdf(uploaded_file):
    reader = pdf.PdfReader(uploaded_file)
    return "".join(page.extract_text() or "" for page in reader.pages)

def extract_text_from_docx(uploaded_file):
    return "\n".join(para.text for para in Document(uploaded_file).paragraphs)

def extract_text_from_txt(uploaded_file):
    return uploaded_file.read().decode("utf-8")

def extract_text_from_file(uploaded_file):
    file_type = uploaded_file.name.split(".")[-1].lower()
    try:
        if file_type == "pdf": return extract_text_from_pdf(uploaded_file)
        elif file_type == "docx": return extract_text_from_docx(uploaded_file)
        elif file_type == "txt": return extract_text_from_txt(uploaded_file)
        else: st.error(f"Unsupported file type: {file_type}")
    except Exception as e: st.error(f"Error: {str(e)}")

# ----- Data Analysis -----
def analyze_skills(text):
    vectorizer = CountVectorizer(ngram_range=(1, 2), stop_words='english')
    skills_matrix = vectorizer.fit_transform([text])
    return pd.DataFrame(
        skills_matrix.toarray(),
        columns=vectorizer.get_feature_names_out()
    ).sum().sort_values(ascending=False).head(15)

def calculate_match_stats(scores):
    scores = np.array(scores)
    return pd.DataFrame({
        'Metric': ['Mean', 'Median', 'Std Dev', 'Max', 'Min'],
        'Value': [np.mean(scores), np.median(scores), np.std(scores), np.max(scores), np.min(scores)]
    })

# ----- Language Support -----
def detect_language(text):
    try: return SUPPORTED_LANGUAGES.get(detect(text), "Unknown")
    except: return "Unknown"

def get_prompt(key, language="en"):
    prompts = {
        "resume_summary": {
            "en": "Provide a detailed summary of the candidate's resume.",
            "hi": "उम्मीदवार के रिज्यूमे का विस्तृत सारांश प्रदान करें।",
            "ru": "Предоставьте подробное резюме кандидата."
        },
        "percentage_match": {
            "en": "Compare the resume with the job description and return only the match percentage out of 100. Format: 'The match is 87%'.",
            "hi": "रिज्यूमे की नौकरी विवरण के साथ तुलना करें और केवल 100 में से मिलान प्रतिशत वापस करें।",
            "ru": "Сравните резюме с описанием вакансии и верните процент соответствия."
        }
        # Add other prompts here...
    }
    return prompts[key].get(language, prompts[key]["en"])

# ----- Gemini Integration -----
def get_gemini_response(job_desc, resume_text, prompt, extract_percent=False, language="en"):
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(f"""
    Job Description: {job_desc}
    Resume: {resume_text}
    Task: {prompt}
    {f"Respond in {SUPPORTED_LANGUAGES[language]}" if language != "en" else ""}
    """)
    return extract_percentage(response.text) if extract_percent else response.text

def extract_percentage(text):
    return re.search(r"(\d+)%", text).group(1) if re.search(r"(\d+)%", text) else "N/A"

# ----- Session State -----
if "resume_text" not in st.session_state: st.session_state.resume_text = ""
if "job_desc" not in st.session_state: st.session_state.job_desc = ""
if "language" not in st.session_state: st.session_state.language = "en"
if "historical_scores" not in st.session_state: st.session_state.historical_scores = list(np.random.randint(50, 95, 10))

# ----- UI Components -----
with st.sidebar:
    selected = option_menu(
        "AI Resume Analyzer",
        ["Home", "Upload Resume", "Job Description", "Analyze"],
        icons=["house", "cloud-upload", "file-text", "bar-chart"],
        default_index=0
    )
    st.session_state.language = st.selectbox(
        "Language",
        options=list(SUPPORTED_LANGUAGES.keys()),
        format_func=lambda x: SUPPORTED_LANGUAGES[x]
    )

# ----- Pages -----
if selected == "Home":
    st.title("👋 AI Resume Analyzer")
    if st.session_state.resume_text and st.session_state.job_desc:
        match = get_gemini_response(
            st.session_state.job_desc, st.session_state.resume_text,
            get_prompt("percentage_match", st.session_state.language),
            extract_percent=True, language=st.session_state.language
        )
        st.session_state.historical_scores.append(int(match))
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div style="background:#d4fc79; padding:20px; border-radius:15px;">
                <h3>📊 Match Score: {match}%</h3>
            </div>
            """, unsafe_allow_html=True)
            st.plotly_chart(px.histogram(
                x=st.session_state.historical_scores,
                title="Historical Score Distribution",
                labels={"x": "Match Percentage"}
            ))
        
        with col2:
            st.subheader("🔠 Top Skills")
            st.bar_chart(analyze_skills(st.session_state.resume_text))
            
    else: st.info("Upload resume and job description")

elif selected == "Upload Resume":
    st.header("📄 Upload Resume")
    if uploaded_file := st.file_uploader("Choose file", type=["pdf", "docx", "txt"]):
        st.session_state.resume_text = extract_text_from_file(uploaded_file)
        st.success(f"✅ Processed ({detect_language(st.session_state.resume_text)})")

elif selected == "Job Description":
    st.header("💼 Job Description")
    if job_desc := st.text_area("Paste here"):
        st.session_state.job_desc = job_desc
        st.success(f"✅ Saved ({detect_language(job_desc)})")

elif selected == "Analyze":
    if not st.session_state.resume_text or not st.session_state.job_desc:
        st.warning("Upload resume and job description first")
    else:
        tab1, tab2, tab3, tab4 = st.tabs(["Summary", "Match", "Skills", "Data"])
        
        with tab1:
            st.write(get_gemini_response(
                st.session_state.job_desc, st.session_state.resume_text,
                get_prompt("resume_summary", st.session_state.language),
                language=st.session_state.language
            ))
        
        with tab2:
            st.dataframe(calculate_match_stats(st.session_state.historical_scores))
        
        with tab3:
            st.plotly_chart(px.bar(
                analyze_skills(st.session_state.resume_text),
                title="Skill Frequency"
            ))
        
        with tab4:
            st.dataframe(pd.DataFrame({
                "Resume Chunks": [st.session_state.resume_text[i:i+200] 
                for i in range(0, min(1000, len(st.session_state.resume_text)), 200)]
            }))