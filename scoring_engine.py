# scoring_engine.py
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nlp_core import extract_experience, normalize_skills, tfidf_vectorize, sbert_vectorize

import nlp_core

# Then you can use its functions:
skills = nlp_core.normalize_skills("Python, SQL, Pandas")
print(skills)

def ats_score(resume_text, jd_text, skill_weight=0.6, exp_weight=0.4):
    """Compute ATS weighted score based on skill match and experience."""
    skills_resume = normalize_skills(resume_text)
    skills_jd = normalize_skills(jd_text)
    
    skill_match_ratio = len(set(skills_resume) & set(skills_jd)) / max(len(skills_jd), 1)
    experience_years = extract_experience(resume_text)
    exp_score = min(experience_years / 10, 1)  # normalize 0-10 years

    score = skill_weight * skill_match_ratio + exp_weight * exp_score
    return round(score * 100, 2)  # 0-100 scale

def semantic_score(resume_text, jd_text):
    """Compute semantic similarity score using S-BERT embeddings."""
    embeddings = sbert_vectorize([resume_text, jd_text])
    sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return round(sim * 100, 2)

def hybrid_score(resume_text, jd_text, ats_w=0.5, sem_w=0.5):
    """Combine ATS and semantic similarity for final score."""
    ats = ats_score(resume_text, jd_text)
    sem = semantic_score(resume_text, jd_text)
    final = ats_w * ats + sem_w * sem
    return round(final, 2)

def rank_candidates(resumes, jd_text):
    """Rank multiple candidates based on hybrid score."""
    ranked = []
    for r in resumes:
        score = hybrid_score(r['text'], jd_text)
        ranked.append({**r, 'score': score})
    ranked.sort(key=lambda x: x['score'], reverse=True)
    return ranked


def predict_role_with_llm(resume_text):
    """Call Mistral API to predict candidate role."""
    # Placeholder
    return "Data Scientist"