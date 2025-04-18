import streamlit as st
from streamlit_option_menu import option_menu
import os
from dotenv import load_dotenv
import PyPDF2 as pdf
from docx import Document
import google.generativeai as genai
import re

# Load env variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Streamlit page config
st.set_page_config(page_title="AI Resume Analyzer", layout="wide")

# ----- File Processing Functions -----
def extract_text_from_pdf(uploaded_file):
    """Extract text from PDF files"""
    reader = pdf.PdfReader(uploaded_file)
    text = ""
    for page in range(len(reader.pages)):
        text += reader.pages[page].extract_text() or ""
    return text

def extract_text_from_docx(uploaded_file):
    """Extract text from Word documents"""
    doc = Document(uploaded_file)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text_from_txt(uploaded_file):
    """Extract text from plain text files"""
    return uploaded_file.read().decode("utf-8")

def extract_text_from_file(uploaded_file):
    """Main function to handle all file types"""
    file_type = uploaded_file.name.split(".")[-1].lower()
    
    try:
        if file_type == "pdf":
            return extract_text_from_pdf(uploaded_file)
        elif file_type == "docx":
            return extract_text_from_docx(uploaded_file)
        elif file_type == "txt":
            return extract_text_from_txt(uploaded_file)
        else:
            st.error(f"Unsupported file type: {file_type}")
            return None
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None

# ----- Gemini Response Function -----
def get_gemini_response(job_desc, resume_text, prompt, extract_percent=False):
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    full_prompt = f"""
    Job Description:
    {job_desc}

    Resume:
    {resume_text}

    Task:
    {prompt}
    """

    try:
        response = model.generate_content(full_prompt)
        if not response.text:
            return "No response generated."

        if extract_percent:
            return extract_percentage(response.text)

        return response.text

    except Exception as e:
        return f"Error: {str(e)}"

# ----- Extract percentage from response -----
def extract_percentage(response_text):
    match = re.search(r"(\d+)%", response_text)
    if match:
        return match.group(1)
    return "N/A"

# ----- Prompts -----
prompts = {
    "resume_summary": "Provide a detailed summary of the candidate's resume.",
    "percentage_match": "Compare the resume with the job description and return only the match percentage out of 100. Format: 'The match is 87%'.",
    "skill_improvement": "Analyze the skills in the resume compared to the job description. Suggest areas of improvement.",
    "missing_keywords": "List important keywords or skills missing from the resume when compared with the job description.",
    "interview_prep": "Generate a list of potential interview questions based on the resume and job description.",
    "personalized_recommendations": "Suggest ways to improve the resume based on the job description.",
    "role_fitment_score": "Give a fitment score out of 100 indicating how well the resume fits the job role.",
    "skill_proficiency": "Evaluate skill proficiency levels in the resume according to the job description."
}

# ----- Session Storage -----
if "resume_text" not in st.session_state:
    st.session_state.resume_text = ""
if "job_desc" not in st.session_state:
    st.session_state.job_desc = ""

# ----- Sidebar Menu -----
with st.sidebar:
    selected = option_menu(
        "AI Resume Analyzer",
        ["Home", "Upload Resume", "Job Description", "Analyze"],
        icons=["house", "cloud-upload", "file-text", "bar-chart"],
        default_index=0
    )

# ----- Pages -----
if selected == "Home":
    st.title("👋 Welcome to AI Resume Analyzer")
    st.write("This tool helps you optimize your resume for any job using Gemini AI.")

    if st.session_state.resume_text and st.session_state.job_desc:
        job_match_percentage = get_gemini_response(
            st.session_state.job_desc, st.session_state.resume_text, 
            prompts["percentage_match"], extract_percent=True
        )
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #d4fc79 0%, #96e6a1 100%);
                    padding: 20px; border-radius: 15px; color: #000; margin-bottom: 20px;">
            <h3>📊 Job Match Score</h3>
            <p>This resume matches <b>{job_match_percentage}%</b> of the job requirements.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Please upload resume and job description first.")

elif selected == "Upload Resume":
    st.header("📄 Upload Your Resume")
    uploaded_file = st.file_uploader("Upload Resume", type=["pdf", "docx", "txt"])
    
    if uploaded_file:
        st.session_state.resume_text = extract_text_from_file(uploaded_file)
        if st.session_state.resume_text:
            st.success("✅ Resume uploaded and processed successfully.")
            with st.expander("View Extracted Text"):
                st.text(st.session_state.resume_text)

elif selected == "Job Description":
    st.header("💼 Paste Job Description")
    job_desc = st.text_area("Paste the job description here:")
    if job_desc:
        st.session_state.job_desc = job_desc
        st.success("✅ Job description saved.")

elif selected == "Analyze":
    st.header("📊 Resume Analysis")

    if not st.session_state.resume_text or not st.session_state.job_desc:
        st.warning("⚠️ Please upload a resume and enter a job description first.")
    else:
        resume = st.session_state.resume_text
        job = st.session_state.job_desc

        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
            "Summary", "Match %", "Missing Keywords", "Skills", 
            "Interview Qs", "Recommendations", "Fitment Score", "Skill Levels"
        ])

        with st.spinner("Analyzing..."):

            with tab1:
                st.subheader("📄 Resume Summary")
                st.write(get_gemini_response(job, resume, prompts["resume_summary"]))

            with tab2:
                st.subheader("📊 Job Match Percentage")
                match = get_gemini_response(job, resume, prompts["percentage_match"], extract_percent=True)
                st.write(f"Match Percentage: {match}%")

            with tab3:
                st.subheader("🔍 Missing Keywords")
                st.write(get_gemini_response(job, resume, prompts["missing_keywords"]))

            with tab4:
                st.subheader("🛠️ Skill Improvement")
                st.write(get_gemini_response(job, resume, prompts["skill_improvement"]))

            with tab5:
                st.subheader("🗣️ Interview Preparation")
                st.write(get_gemini_response(job, resume, prompts["interview_prep"]))

            with tab6:
                st.subheader("🎯 Personalized Recommendations")
                st.write(get_gemini_response(job, resume, prompts["personalized_recommendations"]))

            with tab7:
                st.subheader("✅ Role Fitment Score")
                st.write(get_gemini_response(job, resume, prompts["role_fitment_score"]))

            with tab8:
                st.subheader("📈 Skill Proficiency")
                st.write(get_gemini_response(job, resume, prompts["skill_proficiency"]))