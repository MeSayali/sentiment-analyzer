"""
Generate a realistic social media sentiment dataset for training and demo.
Run once: python dataset/generate_dataset.py
"""
import pandas as pd
import random
from datetime import datetime, timedelta

random.seed(42)

POSITIVE = [
    "Enjoying a beautiful day at the park!",
    "Just got promoted at work, feeling amazing!",
    "This product is absolutely fantastic, highly recommend!",
    "Had the best coffee this morning, life is good ☕",
    "So grateful for my amazing friends and family ❤️",
    "The new update is incredible, love the features!",
    "Just finished a great workout, feeling unstoppable 💪",
    "This movie was breathtaking, must watch!",
    "Loving the new restaurant downtown, food was delicious!",
    "Achieved my fitness goal today, so proud of myself!",
    "This app has completely changed my life for the better",
    "Best customer service I have ever experienced, wow!",
    "Beautiful sunset tonight, nature is truly amazing 🌅",
    "So excited for the upcoming holiday season!",
    "Finally got my dream job, hard work pays off!",
    "The concert was absolutely mind-blowing last night",
    "Feeling motivated and ready to take on the world",
    "This book is so inspiring, could not put it down",
    "Amazing weather today, perfect for a picnic",
    "My team just won the championship, so proud!",
]

NEGATIVE = [
    "This product is terrible, complete waste of money.",
    "Worst customer service ever, never coming back.",
    "So disappointed with this update, ruined everything.",
    "Traffic is absolutely horrible today, stuck for hours.",
    "My order arrived broken, extremely frustrated.",
    "This app keeps crashing, totally useless.",
    "The movie was a complete disaster, do not watch.",
    "Having the worst day ever, nothing is going right.",
    "This company does not care about their customers at all.",
    "So angry about the price increase, totally unfair.",
    "The food was cold and tasteless, very disappointing.",
    "Lost my job today, feeling hopeless and lost.",
    "This feature is so broken, how did this pass QA?",
    "Waiting 3 hours and still no response from support.",
    "The new policy is absolutely outrageous.",
    "Feeling completely overwhelmed and stressed out.",
    "Service has really gone downhill, very disappointing.",
    "This is the worst experience I have ever had.",
    "My account got hacked, I am furious right now.",
    "Nothing works as advertised, total scam.",
]

NEUTRAL = [
    "Just updated my profile picture.",
    "Watching TV tonight.",
    "Going to the grocery store later.",
    "The weather is cloudy today.",
    "Reading a book this afternoon.",
    "Had lunch at the usual place.",
    "Working from home today.",
    "The meeting was rescheduled to Thursday.",
    "Checking my emails in the morning.",
    "Downloaded the new software version.",
    "Just arrived at the airport.",
    "The conference starts at 9am tomorrow.",
    "New phone arrived in the mail today.",
    "Attended the weekly team standup.",
    "The report is due by end of week.",
    "Currently on a call.",
    "The store opens at 8am.",
    "Just posted a new blog article.",
    "Taking a short break right now.",
    "The system update takes about 10 minutes.",
]

PLATFORMS = ["Twitter", "Instagram", "Facebook", "LinkedIn", "Reddit"]
HASHTAG_POOL = {
    "Positive": ["#happy", "#blessed", "#motivated", "#success", "#love", "#grateful", "#winning", "#amazing"],
    "Negative": ["#frustrated", "#disappointed", "#angry", "#fail", "#terrible", "#worst", "#scam", "#broken"],
    "Neutral":  ["#update", "#news", "#info", "#daily", "#work", "#life", "#tech", "#general"],
}
USERS = [f"user_{i:04d}" for i in range(1, 201)]

base_date = datetime(2024, 1, 1)
rows = []

for _ in range(1000):
    sentiment = random.choices(["Positive", "Negative", "Neutral"], weights=[0.45, 0.35, 0.20])[0]
    text_pool = {"Positive": POSITIVE, "Negative": NEGATIVE, "Neutral": NEUTRAL}[sentiment]
    text = random.choice(text_pool)

    platform = random.choice(PLATFORMS)
    user = random.choice(USERS)
    days_offset = random.randint(0, 364)
    hours_offset = random.randint(0, 23)
    timestamp = base_date + timedelta(days=days_offset, hours=hours_offset)

    hashtags_list = random.sample(HASHTAG_POOL[sentiment], k=random.randint(1, 3))
    hashtags = " ".join(hashtags_list)

    if sentiment == "Positive":
        likes = random.randint(10, 500)
        retweets = random.randint(5, 200)
    elif sentiment == "Negative":
        likes = random.randint(1, 150)
        retweets = random.randint(1, 80)
    else:
        likes = random.randint(0, 80)
        retweets = random.randint(0, 30)

    rows.append({
        "Text": text,
        "Sentiment": sentiment,
        "Timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "User": user,
        "Platform": platform,
        "Hashtags": hashtags,
        "Retweets": retweets,
        "Likes": likes,
    })

df = pd.DataFrame(rows)
df.to_csv("dataset/social_media_sentiment.csv", index=False)
print(f"✅ Dataset generated: {len(df)} rows → dataset/social_media_sentiment.csv")
print(df["Sentiment"].value_counts())
