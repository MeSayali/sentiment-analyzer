"""
app.py  —  Flask Backend
Run: python backend/app.py  →  http://127.0.0.1:5000
"""
import os, re, pickle, json, subprocess, sys
from flask import Flask, request, jsonify, render_template

BASE       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR  = os.path.join(BASE, "backend", "model")
FRONT_DIR  = os.path.join(BASE, "frontend")
UPLOAD_DIR = os.path.join(BASE, "dataset")

app = Flask(__name__, template_folder=os.path.join(FRONT_DIR,"templates"),
            static_folder=os.path.join(FRONT_DIR,"static"))
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024   # 16 MB

def load_artifacts():
    with open(os.path.join(MODEL_DIR,"sentiment_model.pkl"),"rb") as f:  model = pickle.load(f)
    with open(os.path.join(MODEL_DIR,"tfidf_vectorizer.pkl"),"rb") as f: vec   = pickle.load(f)
    with open(os.path.join(MODEL_DIR,"stats.json")) as f:                stats = json.load(f)
    return model, vec, stats

MODEL, VECTORIZER, STATS = load_artifacts()

EMOJI  = {"Positive":"😊","Negative":"😞","Neutral":"😐"}
COLOR  = {"Positive":"#10b981","Negative":"#ef4444","Neutral":"#3b82f6"}

def clean_text(t):
    t = str(t).lower()
    t = re.sub(r"http\S+|www\S+","",t)
    t = re.sub(r"@\w+|#\w+","",t)
    t = re.sub(r"[^\x00-\x7F]","",t)
    t = re.sub(r"[^a-z\s]","",t)
    return re.sub(r"\s+"," ",t).strip()

# ── Pages ──────────────────────────────────────────────────────
@app.route("/")           
def home():        return render_template("index.html")
@app.route("/analyze")    
def analyze():     return render_template("analyze.html")
@app.route("/insights")   
def insights():    return render_template("insights.html")
@app.route("/suggestions")
def suggestions(): return render_template("suggestions.html")
@app.route("/about")      
def about():       return render_template("about.html")

# ── API: Predict ───────────────────────────────────────────────
@app.route("/api/predict", methods=["POST","OPTIONS"])
def predict():
    if request.method == "OPTIONS": return "",204
    data  = request.get_json(silent=True) or {}
    text  = data.get("text","").strip()
    if not text: return jsonify({"error":"No text provided"}), 400
    vec   = VECTORIZER.transform([clean_text(text)])
    label = MODEL.predict(vec)[0]
    proba = MODEL.predict_proba(vec)[0]
    conf  = {c: round(float(p)*100,1) for c,p in zip(MODEL.classes_, proba)}
    return jsonify({"sentiment":label,"emoji":EMOJI.get(label,""),"color":COLOR.get(label,"#666"),
                    "confidence":conf,"input_text":text})

# ── API: Insights ──────────────────────────────────────────────
@app.route("/api/insights", methods=["GET"])
def api_insights(): return jsonify(STATS)

# ── API: Health ────────────────────────────────────────────────
@app.route("/api/health", methods=["GET"])
def health(): return jsonify({"status":"ok","accuracy":STATS.get("accuracy")})

# ── API: Upload CSV & retrain ──────────────────────────────────
@app.route("/api/upload", methods=["POST"])
def upload_csv():
    global MODEL, VECTORIZER, STATS
    if "file" not in request.files:
        return jsonify({"error":"No file part"}), 400
    f = request.files["file"]
    if not f.filename.endswith(".csv"):
        return jsonify({"error":"Only .csv files are accepted"}), 400

    save_path = os.path.join(UPLOAD_DIR, "social_media_sentiment.csv")
    f.save(save_path)

    # Run training script
    try:
        result = subprocess.run(
            [sys.executable, os.path.join(BASE,"backend","train_model.py")],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode != 0:
            return jsonify({"error": "Training failed", "details": result.stderr[-800:]}), 500
        MODEL, VECTORIZER, STATS = load_artifacts()
        return jsonify({"success": True, "message": "Dataset uploaded and model retrained!",
                        "accuracy": STATS.get("accuracy"), "records": STATS.get("total_records")})
    except subprocess.TimeoutExpired:
        return jsonify({"error":"Training timed out (>120s)"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── API: Analyze CSV ──────────────────────────────────────────
@app.route("/api/analyze_csv", methods=["GET"])
def analyze_csv():
    import pandas as pd
    csv_path = os.path.join(UPLOAD_DIR, "social_media_sentiment.csv")
    if not os.path.exists(csv_path):
        return jsonify({"error": "CSV file not found"}), 404
    
    try:
        df = pd.read_csv(csv_path)
        if "Text" not in df.columns:
            return jsonify({"error": "CSV must have a 'Text' column"}), 400
        
        results = []
        for idx, row in df.iterrows():
            text = str(row["Text"]).strip()
            if not text:
                continue
            vec = VECTORIZER.transform([clean_text(text)])
            label = MODEL.predict(vec)[0]
            proba = MODEL.predict_proba(vec)[0]
            conf = {c: round(float(p)*100,1) for c,p in zip(MODEL.classes_, proba)}
            results.append({
                "index": int(idx),
                "text": text,
                "predicted_sentiment": label,
                "confidence": conf,
                "emoji": EMOJI.get(label, ""),
                "color": COLOR.get(label, "#666")
            })
        return jsonify({"results": results, "total": len(results)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.after_request
def cors(r):
    r.headers["Access-Control-Allow-Origin"]  = "*"
    r.headers["Access-Control-Allow-Headers"] = "Content-Type"
    r.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return r

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n  🚀  SentimentAI  →  http://0.0.0.0:{port}\n")
    app.run(debug=False, host="0.0.0.0", port=port)
