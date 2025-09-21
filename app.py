import streamlit as st
from pypdf import PdfReader
import docx
from sentence_transformers import SentenceTransformer, util
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
import sqlite3
import pandas as pd
from datetime import datetime

# --- Database Setup ---
def init_db():
    """Initializes the SQLite database and creates the results table if it doesn't exist."""
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
    """Saves a single analysis result to the database."""
    conn = sqlite3.connect('resume_analysis.db')
    c = conn.cursor()
    timestamp = datetime.now()
    c.execute("INSERT INTO results (timestamp, resume_name, jd_name, score, verdict, feedback) VALUES (?, ?, ?, ?, ?, ?)",
              (timestamp, resume_name, jd_name, score, verdict, feedback))
    conn.commit()
    conn.close()

def get_all_results():
    """Fetches all past results from the database, ordered by the most recent."""
    conn = sqlite3.connect('resume_analysis.db')
    df = pd.read_sql_query("SELECT timestamp, resume_name, jd_name, score, verdict FROM results ORDER BY timestamp DESC", conn)
    conn.close()
    return df

init_db()


# --- Model Loading ---
@st.cache_resource
def load_sentence_transformer_model():
    """Loads the Sentence Transformer model."""
    return SentenceTransformer('all-MiniLM-L6-v2')

st_model = load_sentence_transformer_model()