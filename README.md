# 🧠 Social Media Sentiment Analyzer

A complete Machine Learning + Web Application project for classifying social media
comments as **Positive**, **Negative**, or **Neutral**.

---

## 📁 Project Structure

```
sentiment-analyzer/
├── dataset/
│   ├── generate_dataset.py       ← creates synthetic dataset
│   └── social_media_sentiment.csv
├── backend/
│   ├── train_model.py            ← ML training script
│   ├── app.py                    ← Flask API server
│   └── model/
│       ├── sentiment_model.pkl   ← trained Naive Bayes model
│       ├── tfidf_vectorizer.pkl  ← fitted TF-IDF vectorizer
│       └── stats.json            ← dataset statistics for charts
└── frontend/
    ├── templates/
    │   ├── base.html
    │   ├── index.html            ← Home page
    │   ├── analyze.html          ← Analyze text page
    │   ├── insights.html         ← Charts & visualizations
    │   ├── suggestions.html      ← Strategy suggestions
    │   └── about.html            ← About the project
    └── static/
        ├── css/style.css
        └── images/               ← word charts, confusion matrix
```

---

## ⚙️ Setup Instructions

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate dataset (or use your own CSV)

```bash
python dataset/generate_dataset.py
```

> To use **your own CSV**: place it as `dataset/social_media_sentiment.csv`
> and make sure it has columns: `Text`, `Sentiment`, `Platform`, `Likes`, `Retweets`, `Hashtags`, `Timestamp`

### 3. Train the ML model

```bash
python backend/train_model.py
```

This will:
- Clean and vectorize text with TF-IDF
- Train a Multinomial Naive Bayes classifier
- Save `model.pkl` + `vectorizer.pkl`
- Generate word frequency charts and confusion matrix images
- Save dataset statistics to `stats.json`

### 4. Run the Flask server

```bash
python backend/app.py
```

### 5. Open in browser

```
http://127.0.0.1:5000
```

---

## 🌐 Pages

| Page         | URL            | Description                          |
|--------------|----------------|--------------------------------------|
| Home         | `/`            | Project intro, stats, ML pipeline    |
| Analyze      | `/analyze`     | Live text → sentiment prediction     |
| Insights     | `/insights`    | Charts, word clouds, confusion matrix|
| Suggestions  | `/suggestions` | Strategy recommendations             |
| About        | `/about`       | Tech stack, model explanation        |

---

## 🔌 API Endpoints

| Method | Endpoint        | Description                      |
|--------|-----------------|----------------------------------|
| POST   | `/api/predict`  | Predict sentiment for input text |
| GET    | `/api/insights` | Dataset statistics JSON          |
| GET    | `/api/health`   | Server + model health check      |

### Example: `/api/predict`

**Request:**
```json
POST /api/predict
{ "text": "I love this product, absolutely amazing!" }
```

**Response:**
```json
{
  "sentiment": "Positive",
  "emoji": "😊",
  "color": "#00e676",
  "confidence": { "Positive": 94.2, "Negative": 3.1, "Neutral": 2.7 },
  "input_text": "I love this product, absolutely amazing!"
}
```

---

## 🤖 ML Pipeline

```
Raw Text
   ↓ Lowercase, remove URLs/mentions/punctuation
Clean Text
   ↓ TF-IDF Vectorizer (5000 features, unigrams + bigrams)
Feature Matrix
   ↓ Multinomial Naive Bayes (alpha=0.5)
Sentiment Label → Positive / Negative / Neutral
```

---

## 📊 Charts Included

- Sentiment distribution (Donut chart)
- Platform × Sentiment (Grouped bar)
- Avg Likes per Sentiment (Bar)
- Avg Retweets per Sentiment (Bar)
- Sentiment over Time (Line)
- Engagement by Platform (Horizontal bar)
- Word frequency charts per sentiment
- Model Confusion Matrix

---

## 🛠 Technologies

| Layer      | Technology           |
|------------|----------------------|
| ML         | Scikit-learn, Pandas |
| Backend    | Flask (Python)       |
| Frontend   | HTML, CSS, JS        |
| Charts     | Chart.js             |
| Word Viz   | wordcloud / matplotlib|
| Serialization | Pickle            |
