---
title: Resume Relevance Checker
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: "1.36.0"
app_file: app.py
pinned: false
---

# 🤖 Automated Resume Relevance Checker

An intelligent resume analysis system that compares resumes against job descriptions using AI and NLP techniques.

## Features

- 📄 **Multi-format Support**: Upload PDF and DOCX files
- 🎯 **Smart Matching**: Uses both keyword matching and semantic similarity
- 🤖 **AI Feedback**: Powered by Google Gemini for personalized suggestions
- 📊 **Analysis History**: Track all your previous analyses
- 🎨 **Modern UI**: Clean, intuitive interface built with Streamlit

## How It Works

1. **Upload Files**: Add your resume and job description (PDF/DOCX)
2. **AI Analysis**: The system extracts skills and calculates relevance scores
3. **Get Results**: Receive a comprehensive analysis with actionable feedback
4. **Track Progress**: View your analysis history and improvements over time

## Technology Stack

- **Frontend**: Streamlit
- **AI Models**: Google Gemini, Sentence Transformers
- **NLP**: spaCy for text processing
- **Document Processing**: PyMuPDF, python-docx
- **Database**: SQLite for history tracking

## Setup Instructions

To run locally:

```bash
# Clone the repository
git clone <your-repo-url>
cd resume-analyzer

# Install dependencies
pip install -r requirements.txt

# Set your Google API key
export GOOGLE_API_KEY="your-google-api-key"

# Run the app
streamlit run app.py
```

## Environment Variables

- `GOOGLE_API_KEY`: Your Google Gemini API key (required for AI feedback)

## License

MIT

---

*Built with ❤️ and deployed on 🤗 Hugging Face Spaces*
