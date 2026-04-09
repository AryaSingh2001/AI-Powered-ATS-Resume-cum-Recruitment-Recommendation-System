# nlp_core.py
import re
import spacy
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import numpy as np

# Load spaCy model for NER
nlp = spacy.load("en_core_web_sm")

# Load S-BERT model
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

# Sample Data Science skill taxonomy
DATA_SCI_SKILLS = [
    'python', 'r', 'sql', 'pandas', 'numpy', 'scikit-learn', 
    'tensorflow', 'pytorch', 'matplotlib', 'seaborn', 'keras'
]

def extract_entities(text):
    """Extract named entities from resume text."""
    doc = nlp(text)
    entities = {ent.label_: ent.text for ent in doc.ents}
    return entities

def extract_experience(text):
    """Extract years of experience using regex."""
    pattern = r'(\d+)\s+years?'
    matches = re.findall(pattern, text.lower())
    return max([int(m) for m in matches], default=0)

def normalize_skills(text):
    """Normalize and match skills to taxonomy."""
    text = text.lower()
    matched_skills = [skill for skill in DATA_SCI_SKILLS if skill in text]
    return matched_skills

def tfidf_vectorize(corpus):
    """Compute TF-IDF vectors for a list of documents."""
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(corpus)
    return vectorizer, tfidf_matrix

def sbert_vectorize(corpus):
    """Compute S-BERT embeddings for semantic similarity."""
    embeddings = sbert_model.encode(corpus)
    return embeddings