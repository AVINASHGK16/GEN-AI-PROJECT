import streamlit as st
import fitz  # PyMuPDF
import docx
import spacy
from sentence_transformers import SentenceTransformer, util
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

# --- Model Loading ---

@st.cache_resource
def load_spacy_model():
    """Loads the spaCy model."""
    return spacy.load("en_core_web_sm")

@st.cache_resource
def load_sentence_transformer_model():
    """Loads the Sentence Transformer model."""
    return SentenceTransformer('all-MiniLM-L6-v2')

nlp = load_spacy_model()
st_model = load_sentence_transformer_model()

# --- Helper Functions ---

def extract_text_from_file(file):
    """Extracts text from PDF or DOCX file."""
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
    """Extracts skills from text."""
    doc = nlp(text.lower())
    found_skills = set()
    for skill in skill_keywords:
        if skill in text.lower():
            found_skills.add(skill)
    return list(found_skills)

def calculate_semantic_similarity(text1, text2):
    """Calculates semantic similarity."""
    embedding1 = st_model.encode(text1, convert_to_tensor=True)
    embedding2 = st_model.encode(text2, convert_to_tensor=True)
    cosine_scores = util.cos_sim(embedding1, embedding2)
    return cosine_scores.item() * 100

def generate_feedback(jd_text, missing_skills):
    """Generates personalized feedback using an LLM."""
    try:
        # Check if the API key is available in secrets
        if not st.secrets["OPENAI_API_KEY"]:
            return "OpenAI API key not found. Please add it to your secrets file."
            
        llm = ChatOpenAI(api_key=st.secrets["OPENAI_API_KEY"], model_name="gpt-3.5-turbo", temperature=0.5)
        
        feedback_prompt = PromptTemplate(
            input_variables=["job_description", "missing_skills"],
            template="""
            You are an expert career coach providing feedback to a student applying for a job.
            
            Job Description Summary:
            {job_description}
            
            The student's resume is missing the following key skills mentioned in the job description: {missing_skills}.
            
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

# --- Streamlit App Interface ---
st.set_page_config(page_title="Automated Resume Relevance Checker", layout="wide")
st.title("🤖 Automated Resume Relevance Check System")

# Define columns for layout
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
        with st.spinner("Analyzing..."):
            jd_skills = extract_skills(jd_text, SKILL_KEYWORDS)
            resume_skills = extract_skills(resume_text, SKILL_KEYWORDS)
            common_skills = list(set(jd_skills) & set(resume_skills))
            missing_skills = list(set(jd_skills) - set(resume_skills))
            
            hard_match_score = (len(common_skills) / len(jd_skills) * 100) if jd_skills else 0
            semantic_score = calculate_semantic_similarity(jd_text, resume_text)
            
            final_score = (hard_match_score * 0.4) + (semantic_score * 0.6)
            
            st.subheader(f"Final Relevance Score: {int(final_score)}%")
            st.progress(int(final_score))

            if final_score >= 75:
                st.success("Verdict: **High Suitability**")
            elif final_score >= 50:
                st.warning("Verdict: **Medium Suitability**")
            else:
                st.error("Verdict: **Low Suitability**")
            
            st.subheader("Personalized Feedback")
            feedback = generate_feedback(jd_text, missing_skills)
            st.markdown(feedback)

            with st.expander("Show Detailed Analysis"):
                st.markdown(f"**✅ Common Skills ({len(common_skills)}):** `{', '.join(common_skills) if common_skills else 'None'}`")
                st.markdown(f"**❌ Missing Skills ({len(missing_skills)}):** `{', '.join(missing_skills) if missing_skills else 'None'}`")
                st.markdown(f"**Keyword Match Score:** `{int(hard_match_score)}%`")
                st.markdown(f"**Contextual Match Score:** `{int(semantic_score)}%`")
else:
    with col2:
        st.info("Upload a job description and a resume, then click 'Analyze'.")