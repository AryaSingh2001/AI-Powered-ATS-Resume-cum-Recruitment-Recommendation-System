Module 1: NLP Core (Resume Parsing + Embeddings)

We’ll use:

spaCy for NER (Named Entity Recognition)
sklearn for TF-IDF vectorization
sentence-transformers for semantic embeddings (S-BERT)
regex for experience extraction
A data science skill taxonomy for multi-skill classification

Module 2: Scoring Engine

We will compute:

ATS Weighted Score → Skill match + experience + keywords
Cosine Similarity Score → Resume vs JD semantic match
Hybrid Final Score → Weighted combination
Multi-skill ranking → Handle candidates with overlapping domains

Module 3: LLM Integration (Placeholder)

Later, we can plug in Ollama Mistral for:

Role prediction
Multi-skill interconnectivity
Gap analysis

Example placeholder:

Module 4 — API layer: Flask endpoints exposing everything above.

Module 5 — Frontend dashboard: Candidate view (ATS score breakdown, gap analysis) + Recruiter view (ranked candidate table, fit classification).
