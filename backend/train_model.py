"""
train_model.py  —  Sentiment Analyzer Training Script
Handles: Text, Sentiment, Timestamp, User, Platform,
         Hashtags, Retweets, Likes, Country, Year, Month, Day, Hour
Run: python backend/train_model.py
"""

import os, re, pickle, json, random
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

random.seed(42)
np.random.seed(42)

BASE      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE, "dataset", "social_media_sentiment.csv")
MODEL_DIR = os.path.join(BASE, "backend", "model")
IMG_DIR   = os.path.join(BASE, "frontend", "static", "images")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

# ── Color palette ─────────────────────────────────────────────
PALETTE = {
    "Positive": "#10b981", "Negative": "#ef4444", "Neutral": "#3b82f6",
    "bg": "#ffffff", "bg2": "#f8fafc", "border": "#e2e8f0",
    "text": "#0f172a", "text2": "#64748b",
}

plt.rcParams.update({
    "figure.facecolor": PALETTE["bg"], "axes.facecolor": PALETTE["bg"],
    "axes.edgecolor": PALETTE["border"], "axes.labelcolor": PALETTE["text2"],
    "text.color": PALETTE["text"], "xtick.color": PALETTE["text2"],
    "ytick.color": PALETTE["text2"], "grid.color": "#f1f5f9",
    "grid.linestyle": "-", "grid.alpha": 1.0,
    "font.family": "DejaVu Sans", "font.size": 11,
})


# ══════════════════════════════════════════════════════════════
#  1. LOAD & AUGMENT
# ══════════════════════════════════════════════════════════════
print("\n── Loading dataset ──────────────────────────────────────")
df = pd.read_csv(DATA_PATH)
if "Unnamed: 0" in df.columns:
    df.drop(columns=["Unnamed: 0"], inplace=True)

# ── Map sentiments to 3 classes ──────────────────────────────
sentiment_map = {
    'Positive': 'Positive', 'Negative': 'Negative', 'Neutral': 'Neutral',
    'Joy': 'Positive', 'Excitement': 'Positive', 'Contentment': 'Positive',
    'Gratitude': 'Positive', 'Happy': 'Positive', 'Hopeful': 'Positive',
    'Pride': 'Positive', 'Elation': 'Positive', 'Enthusiasm': 'Positive',
    'Euphoria': 'Positive', 'Inspiration': 'Positive', 'Hope': 'Positive',
    'Inspired': 'Positive', 'Empowerment': 'Positive', 'Happiness': 'Positive',
    'Compassion': 'Positive', 'Proud': 'Positive', 'Thrill': 'Positive',
    'Enchantment': 'Positive', 'Reflection': 'Neutral', 'Compassionate': 'Positive',
    'Tenderness': 'Positive', 'Arousal': 'Positive', 'Reverence': 'Positive',
    'Grateful': 'Positive', 'Admiration': 'Positive', 'Calmness': 'Neutral',
    'Fulfillment': 'Positive', 'Anticipation': 'Positive', 'Love': 'Positive',
    'Amusement': 'Positive', 'Accomplishment': 'Positive', 'Satisfaction': 'Positive',
    'Adventure': 'Positive', 'Wonder': 'Positive', 'Harmony': 'Positive',
    'Empathetic': 'Positive', 'Creativity': 'Positive', 'Confident': 'Positive',
    'Free-spirited': 'Positive', 'Zest': 'Positive', 'Enjoyment': 'Positive',
    'Adoration': 'Positive', 'Rejuvenation': 'Positive', 'Resilience': 'Positive',
    'Coziness': 'Positive', 'Exploration': 'Positive', 'Mischievous': 'Neutral',
    'Emotion': 'Neutral', 'Tranquility': 'Neutral', 'Whimsy': 'Positive',
    'Radiance': 'Positive', 'Contemplation': 'Neutral', 'Captivation': 'Positive',
    'FestiveJoy': 'Positive', 'Intrigue': 'Positive', 'Melodic': 'Positive',
    'Optimism': 'Positive', 'Bittersweet': 'Neutral', 'Motivation': 'Positive',
    'JoyfulReunion': 'Positive', 'Overjoyed': 'Positive', 'Wonderment': 'Positive',
    'Appreciation': 'Positive', 'Confidence': 'Positive', 'Blessed': 'Positive',
    'Pensive': 'Neutral', 'Mindfulness': 'Neutral', 'Elegance': 'Positive',
    'Spark': 'Positive', 'Adrenaline': 'Positive', 'ArtisticBurst': 'Positive',
    'CulinaryOdyssey': 'Positive', 'Immersion': 'Positive', 'Freedom': 'Positive',
    'Dazzle': 'Positive', 'InnerJourney': 'Neutral', 'DreamChaser': 'Positive',
    'PlayfulJoy': 'Positive', 'Amazement': 'Positive', 'Success': 'Positive',
    'Friendship': 'Positive', 'Romance': 'Positive', 'Grandeur': 'Positive',
    'Energy': 'Positive', 'Celebration': 'Positive', 'Charm': 'Positive',
    'Ecstasy': 'Positive', 'Colorful': 'Positive', 'Positivity': 'Positive',
    'Solitude': 'Neutral', 'Kindness': 'Positive', 'Connection': 'Positive',
    'Hypnotic': 'Positive', 'Suspense': 'Neutral', 'Triumph': 'Positive',
    'Challenge': 'Neutral', 'Solace': 'Positive', 'Heartwarming': 'Positive',
    'Sympathy': 'Positive', 'Mesmerizing': 'Positive', 'Vibrancy': 'Positive',
    'Imagination': 'Positive', 'Envisioning History': 'Neutral', 'Joy in Baking': 'Positive',
    'Breakthrough': 'Positive', 'Culinary Adventure': 'Positive', 'Winter Magic': 'Positive',
    "Ocean's Freedom": 'Positive', 'Runway Creativity': 'Positive', 'Creative Inspiration': 'Positive',
    'Celestial Wonder': 'Positive', "Nature's Beauty": 'Positive', 'Thrilling Journey': 'Positive',
    'Whispers of the Past': 'Neutral', 'Relief': 'Positive',
    'Despair': 'Negative', 'Nostalgia': 'Neutral', 'Loneliness': 'Negative',
    'Awe': 'Positive', 'Sad': 'Negative', 'Grief': 'Negative',
    'Embarrassed': 'Negative', 'Confusion': 'Negative', 'Acceptance': 'Neutral',
    'Determination': 'Positive', 'Melancholy': 'Negative', 'Frustration': 'Negative',
    'Indifference': 'Neutral', 'Surprise': 'Neutral', 'Ambivalence': 'Neutral',
    'Regret': 'Negative', 'Numbness': 'Negative', 'Playful': 'Positive',
    'Bad': 'Negative', 'Hate': 'Negative', 'Betrayal': 'Negative',
    'Frustrated': 'Negative', 'Bitterness': 'Negative', 'Disgust': 'Negative',
    'Boredom': 'Negative', 'Overwhelmed': 'Negative', 'Heartbreak': 'Negative',
    'Devastated': 'Negative', 'Jealous': 'Negative', 'Jealousy': 'Negative',
    'Resentment': 'Negative', 'Bitter': 'Negative', 'Envious': 'Negative',
    'Fearful': 'Negative', 'Helplessness': 'Negative', 'Intimidation': 'Negative',
    'Anxiety': 'Negative', 'Anger': 'Negative', 'Affection': 'Positive',
    'Fear': 'Negative', 'Sadness': 'Negative', 'Envy': 'Negative',
    'Disappointed': 'Negative', 'Disappointment': 'Negative', 'Sorrow': 'Negative',
    'Loss': 'Negative', 'Apprehensive': 'Negative', 'Yearning': 'Neutral',
    'Isolation': 'Negative', 'Suffering': 'Negative', 'EmotionalStorm': 'Negative',
    'Marvel': 'Positive', 'Heartache': 'Negative', 'Desperation': 'Negative',
    'Darkness': 'Negative', 'Exhaustion': 'Negative', 'Ruins': 'Negative',
    'LostLove': 'Negative', 'Touched': 'Positive', 'Engagement': 'Neutral',
    'Journey': 'Neutral', 'Iconic': 'Positive', 'Obstacle': 'Negative',
    'Pressure': 'Negative', 'Renewed Effort': 'Positive', 'Miscalculation': 'Negative',
    'Shame': 'Negative', 'Dismissive': 'Negative', 'Desolation': 'Negative',
}
df['Sentiment'] = df['Sentiment'].map(sentiment_map).fillna('Neutral')

df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
df["Month"]     = df["Timestamp"].dt.to_period("M").astype(str)
df["Retweets"]  = pd.to_numeric(df.get("Retweets", 0), errors="coerce").fillna(0).astype(int)
df["Likes"]     = pd.to_numeric(df.get("Likes", 0),    errors="coerce").fillna(0).astype(int)
df["Engagement"] = df["Retweets"] + df["Likes"]

has_country  = "Country" in df.columns
has_hour     = "Hour" in df.columns

print(f"  Loaded {len(df)} rows | Columns: {list(df.columns)}")
print(f"  Sentiment: {df['Sentiment'].value_counts().to_dict()}")

# ── Augment to ~500 rows ────────────────────────────────────────
EXTRAS = {
    "Positive": [
        "Best coffee I have had all week, absolutely wonderful morning!",
        "So proud of what our team accomplished today.",
        "Just got back from a beautiful hike, feeling refreshed!",
        "Loving every moment of this vacation.",
        "Finally finished that book I have been reading, amazing ending!",
        "Got promoted today, all the hard work paid off!",
        "Perfect weather for a morning run.",
        "Just adopted a rescue dog, my heart is full.",
        "Our product launch went incredibly well today!",
        "Grateful for supportive friends and family always.",
        "Incredible meal at the new restaurant downtown.",
        "Finished my first marathon, feeling unstoppable!",
        "The concert was absolutely breathtaking last night.",
        "So excited to start this new chapter in my life.",
        "Just hit my savings goal, discipline really pays off!",
        "Amazing workout this morning, feeling energized!",
        "Successfully launched our app with zero bugs today.",
        "Got great feedback from my mentor, so encouraging.",
        "The sunset tonight was genuinely spectacular.",
        "Completed my online certification course finally!",
    ],
    "Negative": [
        "My flight was delayed three hours with zero explanation.",
        "This customer service experience was absolutely dreadful.",
        "My laptop crashed right before the big presentation.",
        "Another pointless meeting that should have been an email.",
        "Prices keep rising but quality keeps dropping.",
        "Stuck in terrible traffic for the third time this week.",
        "The app update broke all my favourite features.",
        "Really disappointed with how this project turned out.",
        "Getting so tired of the constant technical issues at work.",
        "Worst delivery experience I have ever had.",
        "Cannot believe they cancelled my order last minute.",
        "The noise outside is absolutely unbearable tonight.",
        "My subscription was charged twice this billing cycle.",
        "Finding it really hard to stay motivated these days.",
        "Feeling completely overwhelmed with deadlines.",
        "The restaurant forgot half my order and was rude about it.",
        "My package arrived damaged for the second time.",
        "Very frustrated with the lack of communication from support.",
        "Dealing with rude coworkers is absolutely exhausting.",
        "Terrible experience at the hospital today, long wait.",
    ],
    "Neutral": [
        "Checked the weather forecast for the upcoming week.",
        "Attended the regular team standup this morning.",
        "Picked up some groceries on the way home today.",
        "Downloaded the latest software update just now.",
        "The meeting has been rescheduled to Thursday.",
        "Working from home again today.",
        "Replied to emails before lunch.",
        "Set up calendar reminders for next week.",
        "The office supply order arrived this afternoon.",
        "Reading through the quarterly report right now.",
        "Watching the local news this evening.",
        "Updated my LinkedIn profile this afternoon.",
        "The conference call is at nine tomorrow morning.",
        "Reviewed the project timeline with the team.",
        "Stopped by the library to return some books today.",
        "Scheduled my annual health checkup for next month.",
        "The bus was running on time this morning.",
        "Browsing some articles about city infrastructure.",
        "Had a regular Thursday at the office.",
        "Waiting for the technician to arrive.",
    ],
}
PLATFORMS = ["Twitter", "Instagram", "Facebook", "LinkedIn", "Reddit"]
COUNTRIES = ["USA", "UK", "Canada", "Australia", "India", "Germany", "France"]
HASHTAG_MAP = {
    "Positive": ["#HappyVibes", "#Grateful", "#Winning", "#LoveLife", "#Positive", "#Blessed"],
    "Negative": ["#Frustrated", "#Disappointed", "#Rant", "#Fail", "#NotOkay"],
    "Neutral":  ["#Daily", "#Update", "#Work", "#Life", "#Tech", "#Info"],
}
USERS = [f"User{str(i).zfill(3)}" for i in range(1, 200)]

from datetime import datetime, timedelta
base_date = datetime(2023, 1, 1)

existing_counts = df["Sentiment"].value_counts().to_dict()
targets = {"Positive": 200, "Negative": 150, "Neutral": 100}
new_rows = []

for sent, pool in EXTRAS.items():
    needed = max(0, targets[sent] - existing_counts.get(sent, 0))
    for i in range(needed):
        ts = base_date + timedelta(days=random.randint(0, 364), hours=random.randint(0, 23))
        rt = random.randint(5, 30) if sent == "Positive" else random.randint(1, 20)
        lk = random.randint(10, 60) if sent == "Positive" else random.randint(2, 30)
        row = {
            "Text": random.choice(pool), "Sentiment": sent,
            "Timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "User": random.choice(USERS), "Platform": random.choice(PLATFORMS),
            "Hashtags": " ".join(random.sample(HASHTAG_MAP[sent], 2)),
            "Retweets": rt, "Likes": lk, "Engagement": rt + lk,
            "Country": random.choice(COUNTRIES), "Month": ts.strftime("%Y-%m"),
            "Year": ts.year, "Day": ts.day, "Hour": ts.hour,
        }
        new_rows.append(row)

if new_rows:
    df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    df["Engagement"] = df["Retweets"] + df["Likes"]

print(f"\n  After augmentation: {len(df)} rows")
print(f"  {df['Sentiment'].value_counts().to_dict()}")


# ══════════════════════════════════════════════════════════════
#  2. CLEAN TEXT
# ══════════════════════════════════════════════════════════════
def clean_text(t):
    t = str(t).lower()
    t = re.sub(r"http\S+|www\S+", "", t)
    t = re.sub(r"@\w+|#\w+", "", t)
    t = re.sub(r"[^\x00-\x7F]", "", t)
    t = re.sub(r"[^a-z\s]", "", t)
    return re.sub(r"\s+", " ", t).strip()

df["clean_text"] = df["Text"].apply(clean_text)
df = df[df["clean_text"].str.len() > 3].dropna(subset=["Sentiment"])


# ══════════════════════════════════════════════════════════════
#  3. TF-IDF + NAIVE BAYES
# ══════════════════════════════════════════════════════════════
print("\n── Training model ───────────────────────────────────────")
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2),
                              sublinear_tf=True, min_df=1)
X = vectorizer.fit_transform(df["clean_text"])
y = df["Sentiment"]

cv_k = max(2, min(5, y.value_counts().min()))
cv_scores = cross_val_score(MultinomialNB(alpha=0.5), X, y, cv=cv_k)
print(f"  CV ({cv_k}-fold): {cv_scores.mean()*100:.1f}% ± {cv_scores.std()*100:.1f}%")

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = MultinomialNB(alpha=0.5)
model.fit(X_tr, y_tr)
y_pred = model.predict(X_te)
acc = accuracy_score(y_te, y_pred)
print(f"  Test accuracy: {acc*100:.1f}%")
print(classification_report(y_te, y_pred))
model.fit(X, y)   # refit on full data


# ══════════════════════════════════════════════════════════════
#  4. SAVE MODEL + STATS
# ══════════════════════════════════════════════════════════════
with open(os.path.join(MODEL_DIR, "sentiment_model.pkl"),  "wb") as f: pickle.dump(model, f)
with open(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"), "wb") as f: pickle.dump(vectorizer, f)

sc = df["Sentiment"].value_counts().to_dict()
plat_sent = df.groupby(["Platform","Sentiment"]).size().reset_index(name="count").to_dict("records")
rt_sent   = df.groupby("Sentiment")["Retweets"].mean().round(1).to_dict()
lk_sent   = df.groupby("Sentiment")["Likes"].mean().round(1).to_dict()
eng_plat  = df.groupby("Platform")["Engagement"].mean().round(1).to_dict()

all_tags = []
for tags in df["Hashtags"].dropna():
    all_tags.extend(str(tags).split())
top_hashtags = [{"tag": t, "count": c} for t, c in Counter(all_tags).most_common(15)]

time_sent = df.groupby(["Month","Sentiment"]).size().reset_index(name="count").to_dict("records")
top_users = (df.groupby("User")["Engagement"].sum().sort_values(ascending=False)
               .head(8).reset_index().rename(columns={"Engagement":"total_eng"})
               .to_dict("records"))

country_sent = {}
if has_country:
    country_sent = df.groupby(["Country","Sentiment"]).size().reset_index(name="count").to_dict("records")

hour_sent = {}
if has_hour:
    hour_sent = df.groupby(["Hour","Sentiment"]).size().reset_index(name="count").to_dict("records")

stats = {
    "total_records": int(len(df)),
    "accuracy": round(cv_scores.mean()*100, 1),
    "cv_std":   round(cv_scores.std()*100, 1),
    "sentiment_counts": sc,
    "platform_sentiment": plat_sent,
    "avg_retweets_sentiment": rt_sent,
    "avg_likes_sentiment": lk_sent,
    "engagement_by_platform": eng_plat,
    "top_hashtags": top_hashtags,
    "sentiment_over_time": time_sent,
    "top_users": top_users,
    "country_sentiment": country_sent,
    "hour_sentiment": hour_sent,
    "platforms": df["Platform"].unique().tolist(),
    "countries": df["Country"].unique().tolist() if has_country else [],
    "date_range": (lambda ts: {"start": ts.min().strftime("%b %d, %Y") if len(ts) else "", "end": ts.max().strftime("%b %d, %Y") if len(ts) else ""})(pd.to_datetime(df["Timestamp"], errors="coerce").dropna())
}
with open(os.path.join(MODEL_DIR, "stats.json"), "w") as f:
    json.dump(stats, f, indent=2)
print("  Saved: model + vectorizer + stats.json")


# ══════════════════════════════════════════════════════════════
#  5. GENERATE CHARTS
# ══════════════════════════════════════════════════════════════
print("\n── Generating charts ────────────────────────────────────")
SENT_COLORS = {"Positive": "#10b981", "Negative": "#ef4444", "Neutral": "#3b82f6"}
SENTIMENTS  = ["Positive", "Negative", "Neutral"]


def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=130, bbox_inches="tight",
                facecolor=PALETTE["bg"], edgecolor="none")
    plt.close()
    print(f"  Saved: {os.path.basename(path)}")


# ── A. Sentiment Donut ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 6))
vals   = [sc.get(s, 0) for s in SENTIMENTS]
colors = [SENT_COLORS[s] for s in SENTIMENTS]
wedges, texts, autotexts = ax.pie(
    vals, labels=SENTIMENTS, colors=colors, autopct="%1.0f%%",
    startangle=90, pctdistance=0.78,
    wedgeprops=dict(width=0.52, edgecolor="white", linewidth=3))
for t in texts:     t.set(fontsize=12, fontweight="bold")
for t in autotexts: t.set(fontsize=11, color="white", fontweight="bold")
ax.set_title("Sentiment Distribution", fontsize=14, fontweight="bold", pad=20)
savefig(os.path.join(IMG_DIR, "chart_distribution.png"))

# ── B. Platform × Sentiment Grouped Bar ───────────────────────
platforms = df["Platform"].unique()
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(platforms)); w = 0.26
for i, s in enumerate(SENTIMENTS):
    vals = [df[(df["Platform"]==p)&(df["Sentiment"]==s)].shape[0] for p in platforms]
    ax.bar(x+i*w - w, vals, w, label=s, color=SENT_COLORS[s]+"cc",
           edgecolor=SENT_COLORS[s], linewidth=1.2, alpha=0.9, zorder=3)
ax.set_xticks(x); ax.set_xticklabels(platforms)
ax.set_ylabel("Posts"); ax.legend(frameon=False)
ax.set_title("Posts by Platform & Sentiment", fontsize=14, fontweight="bold", pad=14)
ax.yaxis.grid(True, zorder=0); ax.set_axisbelow(True)
ax.spines[["top","right"]].set_visible(False)
savefig(os.path.join(IMG_DIR, "chart_platform.png"))

# ── C. Avg Likes + Retweets side-by-side ──────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
for ax, metric, title in [
    (axes[0], lk_sent, "Avg Likes per Sentiment"),
    (axes[1], rt_sent, "Avg Retweets per Sentiment")
]:
    vals  = [metric.get(s, 0) for s in SENTIMENTS]
    bars  = ax.bar(SENTIMENTS, vals,
                   color=[SENT_COLORS[s]+"cc" for s in SENTIMENTS],
                   edgecolor=[SENT_COLORS[s] for s in SENTIMENTS],
                   linewidth=1.5, width=0.5, zorder=3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                f"{v:.1f}", ha="center", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.yaxis.grid(True, zorder=0); ax.set_axisbelow(True)
    ax.spines[["top","right"]].set_visible(False)
plt.suptitle("Engagement Metrics by Sentiment", fontsize=15, fontweight="bold", y=1.02)
savefig(os.path.join(IMG_DIR, "chart_engagement.png"))

# ── D. Sentiment Timeline ──────────────────────────────────────
ot = df.groupby(["Month","Sentiment"]).size().reset_index(name="count")
months = sorted(ot["Month"].unique())
fig, ax = plt.subplots(figsize=(11, 5))
for s in SENTIMENTS:
    sub  = ot[ot["Sentiment"]==s]
    vals = [sub[sub["Month"]==m]["count"].sum() for m in months]
    ax.plot(range(len(months)), vals, marker="o", ms=5, lw=2.5,
            color=SENT_COLORS[s], label=s, zorder=3)
    ax.fill_between(range(len(months)), vals, alpha=0.07, color=SENT_COLORS[s])
ax.set_xticks(range(len(months)))
ax.set_xticklabels(months, rotation=35, ha="right", fontsize=9)
ax.set_ylabel("Post Count"); ax.legend(frameon=False)
ax.set_title("Sentiment Trend Over Time", fontsize=14, fontweight="bold", pad=14)
ax.yaxis.grid(True, zorder=0); ax.set_axisbelow(True)
ax.spines[["top","right"]].set_visible(False)
savefig(os.path.join(IMG_DIR, "chart_timeline.png"))

# ── E. Country × Sentiment (if available) ─────────────────────
if has_country:
    countries = df["Country"].value_counts().head(6).index.tolist()
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(countries)); w = 0.26
    for i, s in enumerate(SENTIMENTS):
        vals = [df[(df["Country"]==c)&(df["Sentiment"]==s)].shape[0] for c in countries]
        ax.bar(x+i*w-w, vals, w, label=s, color=SENT_COLORS[s]+"cc",
               edgecolor=SENT_COLORS[s], linewidth=1.2, alpha=0.9, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(countries)
    ax.set_ylabel("Posts"); ax.legend(frameon=False)
    ax.set_title("Sentiment by Country", fontsize=14, fontweight="bold", pad=14)
    ax.yaxis.grid(True, zorder=0); ax.set_axisbelow(True)
    ax.spines[["top","right"]].set_visible(False)
    savefig(os.path.join(IMG_DIR, "chart_country.png"))

# ── F. Hour of Day Sentiment (if available) ────────────────────
if has_hour:
    ht = df.groupby(["Hour","Sentiment"]).size().reset_index(name="count")
    hours = sorted(ht["Hour"].unique())
    fig, ax = plt.subplots(figsize=(11, 5))
    for s in SENTIMENTS:
        sub  = ht[ht["Sentiment"]==s]
        vals = [sub[sub["Hour"]==h]["count"].sum() for h in hours]
        ax.plot(hours, vals, marker="o", ms=4, lw=2,
                color=SENT_COLORS[s], label=s, zorder=3)
    ax.set_xlabel("Hour of Day (0-23)"); ax.set_ylabel("Posts")
    ax.legend(frameon=False)
    ax.set_title("Posting Activity by Hour", fontsize=14, fontweight="bold", pad=14)
    ax.yaxis.grid(True, zorder=0); ax.set_axisbelow(True)
    ax.spines[["top","right"]].set_visible(False)
    savefig(os.path.join(IMG_DIR, "chart_hourly.png"))

# ── G. Confusion Matrix ────────────────────────────────────────
cm = confusion_matrix(y_te, y_pred, labels=SENTIMENTS)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d",
            cmap=sns.light_palette("#6366f1", as_cmap=True),
            xticklabels=SENTIMENTS, yticklabels=SENTIMENTS,
            linewidths=2, linecolor="white", ax=ax,
            annot_kws={"size": 14, "weight": "bold"})
ax.set_title("Confusion Matrix (Test Set)", fontsize=14, fontweight="bold", pad=14)
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
savefig(os.path.join(IMG_DIR, "confusion_matrix.png"))

# ── H. WORDCLOUD images (matplotlib-based) ────────────────────
STOPWORDS = {
    "just","the","is","a","and","to","it","was","this","so","at","my","for","of",
    "i","have","had","are","be","with","not","its","that","we","they","in","got",
    "day","im","ive","about","new","feel","really","today","going","been","some",
    "get","all","time","after","good","one","want","now","still","little","on",
    "an","do","but","up","out","when","how","no","he","she","they","their","our",
    "what","will","can","more","very","only","your","from","has","her","his",
}

def make_wordcloud(words_counts, bg_color, colormap, title, filepath):
    """Generate a professional word cloud using WordCloud library."""
    from wordcloud import WordCloud
    
    # Create word frequency dictionary
    word_freq = {word: freq for word, freq in words_counts}
    
    # Create word cloud
    wc = WordCloud(
        width=1200,
        height=600,
        background_color=bg_color,
        colormap=colormap[0] if isinstance(colormap, list) else colormap,
        max_words=100,
        min_font_size=12,
        max_font_size=100,
        random_state=42,
        collocations=False,
        stopwords=set(),
        contour_width=0,
        prefer_horizontal=0.9,
        mask=None,
        font_path=None
    ).generate_from_frequencies(word_freq)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    
    # Add title with better styling
    title_color = wc.colors[0] if hasattr(wc, 'colors') and wc.colors else '#333333'
    ax.set_title(title, fontsize=18, fontweight='bold', 
                color=title_color, pad=20, loc='center')
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor=bg_color)
    plt.close()
    print(f"  WordCloud saved: {os.path.basename(filepath)}")


WC_THEMES = {
    "Positive": ("#f0fdf4", "Greens"),
    "Negative": ("#fff5f5", "Reds"), 
    "Neutral":  ("#eff6ff", "Blues"),
}

for s in SENTIMENTS:
    subset  = df[df["Sentiment"]==s]["clean_text"]
    counter = Counter()
    for txt in subset:
        for w in txt.split():
            if w not in STOPWORDS and len(w) > 3:
                counter[w] += 1
    top_words = counter.most_common(60)
    if not top_words:
        continue
    bg, cmap = WC_THEMES[s]
    make_wordcloud(
        top_words, bg, [cmap],  # Pass colormap name
        f"Most Common Words — {s} Posts",
        os.path.join(IMG_DIR, f"wordcloud_{s.lower()}.png")
    )

# Combined wordcloud
all_counter = Counter()
for txt in df["clean_text"]:
    for w in txt.split():
        if w not in STOPWORDS and len(w) > 3:
            all_counter[w] += 1
make_wordcloud(
    all_counter.most_common(80),
    "#fafafa",
    ["viridis"],  # Use viridis colormap for all words
    "Most Common Words — All Posts",
    os.path.join(IMG_DIR, "wordcloud_all.png")
)

print(f"\n{'═'*54}")
print(f"  ✅  Done!  CV Accuracy = {cv_scores.mean()*100:.1f}%  |  {len(df)} rows")
print(f"{'═'*54}\n")
