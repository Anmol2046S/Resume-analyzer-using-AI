from dotenv import load_dotenv
load_dotenv()
import streamlit as st
import os
import PyPDF2 as pdf
import google.generativeai as genai
import openai
import anthropic
import cohere
from transformers import pipeline

# Configure APIs
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
openai.api_key = os.getenv("OPENAI_API_KEY")
claude_client = anthropic.Anthropic(api_key=os.getenv("CLAUDE_API_KEY"))
cohere_client = cohere.Client(os.getenv("COHERE_API_KEY"))
#hf_summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Extract text from PDF
def input_pdf_text(uploaded_file):
    reader = pdf.PdfReader(uploaded_file)
    text = ""
    for page in range(len(reader.pages)):
        text += reader.pages[page].extract_text() or ""
    return text

# AI responses
def get_gemini_response(input_text, pdf_content, prompt):
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content([input_text, pdf_content, prompt])
    return response.text

def get_gpt_response(input_text, pdf_content, prompt):
    full_prompt = f"{input_text}\n\n{pdf_content}\n\n{prompt}"
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": full_prompt}],
            temperature=0.7,
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error: {str(e)}"

def get_claude_response(input_text, pdf_content, prompt):
    response = claude_client.messages.create(
        model="claude-3-opus-20240229",
        messages=[{"role": "user", "content": f"{input_text}\n\n{pdf_content}\n\n{prompt}"}],
        max_tokens=1024
    )
    return response.content[0].text

def get_cohere_response(input_text, pdf_content, prompt):
    response = cohere_client.generate(
        model='command-r',
        prompt=f"{input_text}\n\n{pdf_content}\n\n{prompt}",
        max_tokens=300
    )
    return response.generations[0].text

def get_hf_summary(pdf_content):
    return hf_summarizer(pdf_content[:1000])[0]['summary_text']

# Streamlit UI Setup
st.set_page_config(page_title="Multi-AI Resume Analyzer")
st.header("🧠 Resume Analyzer: Compare Multiple AI Tools")

input_text = st.text_area("Paste Job Description:", key="input")
uploaded_file = st.file_uploader("Upload Resume (PDF Only):", type=["pdf"])
ai_choice = st.selectbox("Choose AI Tool to Analyze Resume", ["Gemini", "ChatGPT", "Claude", "Cohere", "HuggingFace"])

if uploaded_file is not None:
    st.success("✅ Resume Uploaded Successfully!")

# Prompts
def get_prompts():
    return {
        "resume_summary": """
        Provide a detailed summary of the resume, highlighting key qualifications, experiences, and notable achievements.
        """,
        "percentage_match": """
        Compare the resume with the job description and provide a percentage match score.
        Explain how well the resume fits the job role based on skills, experience, and keywords.
        """,
        "skill_improvement": """
        Analyze the resume against the provided job description. Identify relevant technical and soft skills.
        For each skill in the job description:
        - If the skill is present in the resume, mark it as ✅.
        - If the skill is missing, mark it as ❌.
        Provide a structured table format for easy readability.
        """,
        "missing_keywords": """
        Extract key skills from the job description and compare them with the resume.
        Identify missing or weak keywords that could improve ATS optimization.
        """,
        "interview_prep": """
        Based on the resume and job description, generate a list of potential interview questions.
        Include both technical and behavioral questions.
        """,
        "personalized_recommendations": """
        Based on the resume and job description, provide personalized recommendations for improving the resume.
        Suggest relevant courses, certifications, and projects that could enhance the candidate's profile.
        """,
        "role_fitment_score": """
        Provide a percentage score (out of 100) on how well the resume aligns with the job description. Consider the match between the skills, experience, and qualifications required for the role.
        """,
        "skill_proficiency": """
        Based on the resume and job description, evaluate the proficiency level of each skill (basic, intermediate, advanced). Provide feedback on how the candidate can improve their proficiency in each skill.
        """
    }

prompts = get_prompts()

# Generic Section Generator
def generate_ai_response(title, prompt_key):
    if st.button(title):
        if uploaded_file is not None and ("job description" not in title.lower() or input_text):
            pdf_content = input_pdf_text(uploaded_file)
            prompt = prompts[prompt_key]

            with st.spinner(f"Generating {title} with {ai_choice}..."):
                if ai_choice == "Gemini":
                    response = get_gemini_response(input_text, pdf_content, prompt)
                elif ai_choice == "ChatGPT":
                    response = get_gpt_response(input_text, pdf_content, prompt)
                elif ai_choice == "Claude":
                    response = get_claude_response(input_text, pdf_content, prompt)
                elif ai_choice == "Cohere":
                    response = get_cohere_response(input_text, pdf_content, prompt)
                elif ai_choice == "HuggingFace":
                    response = get_hf_summary(pdf_content)
                else:
                    response = "❌ Invalid AI Selection"

            st.subheader(f"{ai_choice} Response for {title}:")
            st.write(response)
        else:
            st.warning("⚠️ Please upload a resume and provide a job description.")

# Buttons for All Prompts
generate_ai_response("Summarize my Resume", "resume_summary")
generate_ai_response("Percentage Match", "percentage_match")
generate_ai_response("How can I improve my skills?", "skill_improvement")
generate_ai_response("Find Missing Keywords", "missing_keywords")
generate_ai_response("Interview Preparation", "interview_prep")
generate_ai_response("Personalized Recommendations", "personalized_recommendations")
generate_ai_response("Job Role Fitment Score", "role_fitment_score")
generate_ai_response("Skill Proficiency Levels", "skill_proficiency")