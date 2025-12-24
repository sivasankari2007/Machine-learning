import streamlit as st
import numpy as np
import pickle
import random
import requests
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Hugging Face API
HF_API_TOKEN = "PASTE_YOUR_TOKEN_HERE"
HF_MODEL = "google/flan-t5-base"

headers = {
    "Authorization": f"Bearer {HF_API_TOKEN}"
}

# Load model and tools
model = load_model("chatbot/chatbot_rnn_model.h5")

with open("model/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("model/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

max_len = 20

# Responses
responses = {
    "greeting": ["Hello!", "Hi 😊"],
    "name": ["I am an AI chatbot"],
    "goodbye": ["Bye 👋"]
}

def ask_huggingface(question):
    payload = {"inputs": question}
    response = requests.post(
        f"https://api-inference.huggingface.co/models/{HF_MODEL}",
        headers=headers,
        json=payload
    )
    result = response.json()
    return result[0]["generated_text"]

def chatbot_reply(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len)
    prediction = model.predict(padded)
    confidence = np.max(prediction)

    if confidence > 0.6:
        tag = label_encoder.inverse_transform([np.argmax(prediction)])[0]
        return random.choice(responses[tag])
    else:
        return ask_huggingface(text)

# Streamlit UI
st.title("🤖 AI Chatbot (RNN + Hugging Face)")

user_input = st.text_input("Ask me anything:")

if user_input:
    reply = chatbot_reply(user_input)
    st.write("Bot:", reply)
