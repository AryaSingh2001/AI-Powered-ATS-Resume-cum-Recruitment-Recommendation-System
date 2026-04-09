# app.py
from flask import Flask, request, jsonify
from scoring_engine import rank_candidates

app = Flask(__name__)

@app.route("/rank", methods=["POST"])
def rank_endpoint():
    data = request.json
    resumes = data.get("resumes", [])
    jd = data.get("job_description", "")
    
    ranked = rank_candidates(resumes, jd)
    return jsonify(ranked)

if __name__ == "__main__":
    app.run(debug=True)