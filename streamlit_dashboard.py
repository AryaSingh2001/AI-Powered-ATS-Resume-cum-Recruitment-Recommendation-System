# streamlit_dashboard.py
import streamlit as st
import pandas as pd
from scoring_engine import rank_candidates
from nlp_core import extract_entities, normalize_skills, extract_experience

# --- Streamlit Page Setup ---
st.set_page_config(page_title="AI Recruitment Dashboard", layout="wide")

st.title("AI-Powered Recruitment Dashboard")
st.write("Upload resumes and job description to see ATS scoring, skill gaps, and candidate ranking.")

# --- Job Description Input ---
jd_text = st.text_area("Enter Job Description", height=150)

# --- Resume Upload ---
uploaded_files = st.file_uploader("Upload Resumes (TXT or PDF)", accept_multiple_files=True, type=['txt', 'pdf'])

resumes = []

if uploaded_files:
    for file in uploaded_files:
        if file.type == "text/plain":
            text = file.read().decode("utf-8")
        else:
            # For PDF, use PyPDF2
            import PyPDF2
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
        resumes.append({"filename": file.name, "text": text})

# --- Process & Rank Candidates ---
if st.button("Process & Rank Candidates") and jd_text and resumes:
    ranked_candidates = rank_candidates(resumes, jd_text)
    
    st.subheader("Ranked Candidates")
    
    # Build DataFrame for display
    data = []
    for r in ranked_candidates:
        skills = normalize_skills(r['text'])
        exp = extract_experience(r['text'])
        entities = extract_entities(r['text'])
        data.append({
            "Candidate": r['filename'],
            "Hybrid Score": r['score'],
            "Skills Matched": ", ".join(skills),
            "Years of Experience": exp,
            "Entities Found": str(entities)
        })
    
    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True)
    
    # Optional: Skill Gap Analysis
    st.subheader("Skill Gap Analysis")
    all_skills = set()
    for r in resumes:
        all_skills.update(normalize_skills(r['text']))
    
    jd_skills = set(normalize_skills(jd_text))
    missing_skills = jd_skills - all_skills
    
    st.write(f"Skills required by JD but missing in candidate pool: {', '.join(missing_skills) if missing_skills else 'None'}")
    
    st.success("Processing Complete!")