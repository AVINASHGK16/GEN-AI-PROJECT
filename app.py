import streamlit as st
import fitz  # PyMuPDF
import docx
import spacy
from sentence_transformers import SentenceTransformer, util
#
# --- CHANGE 1: Swapped OpenAI with Google Generative AI ---
from langchain_google_genai import ChatGoogleGenerativeAI
#
from langchain.prompts import PromptTemplate
import sqlite3
import pandas as pd
from datetime import datetime
import io
# --- Database Setup (No changes) ---
def init_db():
    conn = sqlite3.connect('resume_analysis.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY,
            timestamp DATETIME,
            resume_name TEXT,
            jd_name TEXT,
            score INTEGER,
            verdict TEXT,
            feedback TEXT
        )
    ''')
    conn.commit()
    conn.close()

def save_result(resume_name, jd_name, score, verdict, feedback):
    conn = sqlite3.connect('resume_analysis.db')
    c = conn.cursor()
    timestamp = datetime.now()
    c.execute("INSERT INTO results (timestamp, resume_name, jd_name, score, verdict, feedback) VALUES (?, ?, ?, ?, ?, ?)",
              (timestamp, resume_name, jd_name, score, verdict, feedback))
    conn.commit()
    conn.close()

def get_all_results():
    conn = sqlite3.connect('resume_analysis.db')
    df = pd.read_sql_query("SELECT timestamp, resume_name, jd_name, score, verdict FROM results ORDER BY timestamp DESC", conn)
    conn.close()
    return df

init_db()



# --- Model Loading (NEW and IMPROVED) ---

# We are no longer using @st.cache_resource for the spaCy model
# to allow for the download check.

MODEL_NAME = "en_core_web_sm"

def download_spacy_model():
    """Downloads the spaCy model if it's not already installed."""
    try:
        spacy.load(MODEL_NAME)
    except OSError:
        st.info(f"Downloading spaCy model ({MODEL_NAME})... This may take a moment.")
        spacy.cli.download(MODEL_NAME)
        st.rerun() # Rerun the app to load the model after download

download_spacy_model()
nlp = spacy.load(MODEL_NAME)

@st.cache_resource
def load_sentence_transformer_model():
    """Loads the Sentence Transformer model."""
    return SentenceTransformer('all-MiniLM-L6-v2')

st_model = load_sentence_transformer_model()

# --- The rest of your app.py code remains exactly the same ---

# --- Helper Functions ---
def extract_text_from_file(file):
    if file.type == "application/pdf":
        text = ""
        with fitz.open(stream=file.read(), filetype="pdf") as doc:
            for page in doc:
                text += page.get_text()
        return text
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(file)
        return "\n".join([para.text for para in doc.paragraphs])
    return ""

def extract_skills(text, skill_keywords):
    doc = nlp(text.lower())
    found_skills = set()
    for skill in skill_keywords:
        if skill in text.lower():
            found_skills.add(skill)
    return list(found_skills)

def calculate_semantic_similarity(text1, text2):
    embedding1 = st_model.encode(text1, convert_to_tensor=True)
    embedding2 = st_model.encode(text2, convert_to_tensor=True)
    cosine_scores = util.cos_sim(embedding1, embedding2)
    return cosine_scores.item() * 100

#
# --- CHANGE 2: Updated function to use Google Gemini ---
def generate_feedback(jd_text, missing_skills):
    """Generates personalized feedback using Google Gemini."""
    try:
        if "GOOGLE_API_KEY" not in st.secrets or not st.secrets["GOOGLE_API_KEY"]:
            return "Google API key not found. Please add it to your Streamlit secrets."
            
        # Initialize the Google Gemini model instead of ChatOpenAI
        llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=st.secrets["GOOGLE_API_KEY"], temperature=0.5)
        
        # The prompt and the rest of the logic remain the same
        feedback_prompt = PromptTemplate(
            input_variables=["job_description", "missing_skills"],
            template="""
            You are an expert career coach providing feedback to a student applying for a job.
            Job Description Summary: {job_description}
            The student's resume is missing the following key skills: {missing_skills}.
            Please provide a short, encouraging paragraph of personalized feedback.
            Suggest 1-2 practical ways the student could gain experience in these missing areas (e.g., online courses, personal projects).
            Keep the tone positive and constructive.
            """
        )
        
        chain = feedback_prompt | llm
        feedback = chain.invoke({"job_description": jd_text[:2000], "missing_skills": ", ".join(missing_skills)})
        return feedback.content
    except Exception as e:
        return f"Could not generate feedback due to an error: {e}"

# --- Streamlit App Interface (No changes) ---
st.set_page_config(page_title="Automated Resume Relevance Checker", layout="wide")
st.title("🤖 Automated Resume Relevance Check System")

col1, col2 = st.columns([1, 2])

with col1:
    st.header("Inputs")
    jd_file = st.file_uploader("Upload Job Description (PDF/DOCX)", type=["pdf", "docx"])
    resume_file = st.file_uploader("Upload Resume (PDF/DOCX)", type=["pdf", "docx"])
    analyze_button = st.button("Analyze Resume ✨", type="primary")

SKILL_KEYWORDS = [
    'python', 'java', 'c++', 'sql', 'javascript', 'react', 'vue', 'angular',
    'machine learning', 'data analysis', 'pandas', 'numpy', 'scikit-learn', 'tensorflow', 'pytorch',
    'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'git', 'jira', 'agile', 'scrum'
]

if analyze_button and jd_file and resume_file:
    jd_text = extract_text_from_file(jd_file)
    resume_text = extract_text_from_file(resume_file)
    
    with col2:
        st.header("Analysis Results")
        with st.spinner("Analyzing... This may take a moment."):
            jd_skills = extract_skills(jd_text, SKILL_KEYWORDS)
            resume_skills = extract_skills(resume_text, SKILL_KEYWORDS)
            common_skills = list(set(jd_skills) & set(resume_skills))
            missing_skills = list(set(jd_skills) - set(resume_skills))
            hard_match_score = (len(common_skills) / len(jd_skills) * 100) if jd_skills else 0
            semantic_score = calculate_semantic_similarity(jd_text, resume_text)
            final_score = int((hard_match_score * 0.4) + (semantic_score * 0.6))
            
            if final_score >= 75: 
                verdict = "High Suitability"
                st.success(f"Verdict: **{verdict}**")
            elif final_score >= 50: 
                verdict = "Medium Suitability"
                st.warning(f"Verdict: **{verdict}**")
            else: 
                verdict = "Low Suitability"
                st.error(f"Verdict: **{verdict}**")

            st.subheader(f"Final Relevance Score: {final_score}%")
            st.progress(final_score)
            
            st.subheader("Personalized Feedback")
            if missing_skills:
                feedback = generate_feedback(jd_text, missing_skills)
            else:
                feedback = "Excellent! The resume appears to contain all the key skills from the job description."
            st.markdown(feedback)
            
            save_result(resume_file.name, jd_file.name, final_score, verdict, feedback)

            with st.expander("Show Detailed Analysis"):
                st.markdown(f"**✅ Common Skills ({len(common_skills)}):** `{', '.join(common_skills) if common_skills else 'None'}`")
                st.markdown(f"**❌ Missing Skills ({len(missing_skills)}):** `{', '.join(missing_skills) if missing_skills else 'None'}`")
                st.markdown(f"**Keyword Match Score:** `{int(hard_match_score)}%`")
                st.markdown(f"**Contextual Match Score:** `{int(semantic_score)}%`")
else:
    with col2:
        st.info("Upload a job description and a resume, then click 'Analyze'.")

st.header("Submission History")
st.dataframe(get_all_results(), use_container_width=True)