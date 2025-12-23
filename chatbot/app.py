import os
import pickle
import random
import numpy as np
import streamlit as st
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import requests

# --------------------
# Paths
# --------------------
MODEL_DIR = "model"
DATA_DIR = "data"
MAX_LEN = 10

# --------------------
# Load model
# --------------------
model = load_model(os.path.join(MODEL_DIR, "chatbot_rnn_model.h5"))

with open(os.path.join(MODEL_DIR, "tokenizer.pkl"), "rb") as f:
    tokenizer = pickle.load(f)

categories = ["joke", "motivation", "study", "greeting", "text"]

# --------------------
# Load text data
# --------------------
def load_text(filename):
    with open(os.path.join(DATA_DIR, filename), "r", encoding="utf-8") as f:
        return f.read().splitlines()

data_files = {
    "joke": load_text("joke.txt"),
    "motivation": load_text("motivation.txt"),
    "study": load_text("studytips.txt"),
    "greeting": load_text("greetings.txt"),
    "text": load_text("text.txt")
}

# --------------------
# Free APIs
# --------------------
def get_joke():
    try:
        r = requests.get("https://official-joke-api.appspot.com/random_joke", timeout=5)
        j = r.json()
        return j["setup"] + " 😂 " + j["punchline"]
    except:
        return random.choice(data_files["joke"])

def get_motivation():
    try:
        r = requests.get("https://api.quotable.io/random", timeout=5)
        q = r.json()
        return q["content"]
    except:
        return random.choice(data_files["motivation"])

# --------------------
# Chatbot logic
# --------------------
def chatbot_reply(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN)
    pred = np.argmax(model.predict(padded, verbose=0))
    category = categories[pred]

    if category == "joke":
        return get_joke()
    elif category == "motivation":
        return get_motivation()
    else:
        return random.choice(data_files[category])

# --------------------
# Streamlit UI
# --------------------
st.title("🤖 RNN Chatbot")

user_input = st.text_input("You:")

if user_input:
    st.write("Bot:", chatbot_reply(user_input))
