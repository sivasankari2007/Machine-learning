import os
import pickle
import random
import numpy as np
import requests
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -----------------------------
# PATHS
# -----------------------------
MODEL_DIR = "model"
DATA_DIR = "data"
MAX_LEN = 10   # SAME as training

# -----------------------------
# LOAD MODEL
# -----------------------------
model = load_model(os.path.join(MODEL_DIR, "chatbot_rnn_model.h5"))

with open(os.path.join(MODEL_DIR, "tokenizer.pkl"), "rb") as f:
    tokenizer = pickle.load(f)

# SAME ORDER USED IN TRAINING
categories = ["joke", "motivation", "study", "greeting", "text"]

# -----------------------------
# LOAD TEXT FILES
# -----------------------------
def load_text(filename):
    with open(os.path.join(DATA_DIR, filename), "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

responses = {
    "joke": load_text("chatbot/data-20251221T130125Z-1-001/data/joke.txt"),
    "motivation": load_text("chatbot/data-20251221T130125Z-1-001/data/motivation.txt"),
    "study": load_text("chatbot/data-20251221T130125Z-1-001/data/studytips.txt"),
    "greeting": load_text("chatbot/data-20251221T130125Z-1-001/data/greetings.txt"),
    "text": load_text("chatbot/data-20251221T130125Z-1-001/data/text.txt")
}

# -----------------------------
# LIVE API FUNCTIONS
# -----------------------------
def get_live_joke():
    try:
        r = requests.get(
            "https://official-joke-api.appspot.com/random_joke",
            timeout=5
        )
        data = r.json()
        return data["setup"] + " 😂 " + data["punchline"]
    except:
        return random.choice(responses["joke"])

def get_live_motivation():
    try:
        r = requests.get("https://api.quotable.io/random", timeout=5)
        data = r.json()
        return "🌟 " + data["content"] + " — " + data["author"]
    except:
        return random.choice(responses["motivation"])

# -----------------------------
# CHATBOT LOGIC
# -----------------------------
def chatbot_reply(user_text):
    seq = tokenizer.texts_to_sequences([user_text])
    padded = pad_sequences(seq, maxlen=MAX_LEN)

    prediction = model.predict(padded, verbose=0)
    category = categories[np.argmax(prediction)]

    if category == "joke":
        return get_live_joke()
    elif category == "motivation":
        return get_live_motivation()
    else:
        return random.choice(responses[category])

# -----------------------------
# CHAT LOOP
# -----------------------------
print("🤖 Chatbot is running (type 'exit' to stop)")
print("-------------------------------------------")

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("🤖 Goodbye!")
        break

    print("Bot:", chatbot_reply(user_input))

