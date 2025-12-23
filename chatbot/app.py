import os
import pickle
import random
import numpy as np
import requests
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

# --------------------
# FOLDERS
# --------------------
MODEL_DIR = "model"
DATA_DIR = "data"
MAX_LEN = 10

# --------------------
# LOAD MODEL
# --------------------
model = load_model(os.path.join(MODEL_DIR, "chatbot_rnn_model.h5"))

with open(os.path.join(MODEL_DIR, "tokenizer.pkl"), "rb") as f:
    tokenizer = pickle.load(f)

# SAME ORDER AS TRAINING
categories = ["joke", "motivation", "study", "greeting", "text"]

# --------------------
# LOAD TEXT FILES
# --------------------
def load_text(filename):
    with open(os.path.join(DATA_DIR, filename), "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

responses = {
    "joke": load_text("joke.txt"),
    "motivation": load_text("motivation.txt"),
    "study": load_text("studytips.txt"),
    "greeting": load_text("greetings.txt"),
    "text": load_text("text.txt")
}

# --------------------
# FREE API (VERY SIMPLE)
# --------------------
def get_live_joke():
    try:
        r = requests.get("https://official-joke-api.appspot.com/random_joke", timeout=5)
        j = r.json()
        return j["setup"] + " 😂 " + j["punchline"]
    except:
        return random.choice(responses["joke"])

def get_live_motivation():
    try:
        r = requests.get("https://api.quotable.io/random", timeout=5)
        q = r.json()
        return q["content"]
    except:
        return random.choice(responses["motivation"])

# --------------------
# CHATBOT BRAIN
# --------------------
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

# --------------------
# CHAT LOOP
# --------------------
print("🤖 Chatbot started (type exit to stop)")
print("-------------------------------------")

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("🤖 Bye!")
        break
    print("Bot:", chatbot_reply(user_input))
