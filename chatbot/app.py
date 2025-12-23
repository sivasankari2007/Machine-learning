import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np
import random
import os

# --- GOOGLE THEME SETUP ---
st.set_page_config(page_title="Google Chatbot", page_icon="🔍")

st.markdown("""
    <style>
    .main-header { font-family: 'Product Sans', sans-serif; text-align: center; }
    .g-blue { color: #4285F4; font-size: 50px; font-weight: bold; }
    .g-red { color: #EA4335; font-size: 50px; font-weight: bold; }
    .g-yellow { color: #FBBC05; font-size: 50px; font-weight: bold; }
    .g-green { color: #34A853; font-size: 50px; font-weight: bold; }
    .sub-text { color: #5f6368; font-size: 20px; text-align: center; margin-top: -20px; }
    </style>
    <div class='main-header'>
        <span class='g-blue'>G</span><span class='g-red'>o</span><span class='g-yellow'>o</span><span class='g-blue'>g</span><span class='g-green'>l</span><span class='g-red'>e</span>
        <span style='color: #5f6368; font-size: 40px;'>Chatbot</span>
    </div>
    <div class='sub-text'>RNN Machine Learning Project</div>
    """, unsafe_allow_html=True)

# --- CORRECT FILE PATHING ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'chatbot_rnn_model.h5')
TOKEN_PATH = os.path.join(BASE_DIR, 'model', 'tokenizer.pkl')

@st.cache_resource
def load_assets():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(TOKEN_PATH):
        return None, None
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(TOKEN_PATH, 'rb') as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_assets()

# --- ERROR HANDLING ---
if model is None:
    st.error("⚠️ Model files not found! Ensure the 'model' folder contains 'chatbot_rnn_model.h5' and 'tokenizer.pkl'.")
    st.stop()

def get_response(idx):
    # Match this EXACTLY to your training labels
    mapping = {0: "greetings", 1: "joke", 2: "motivation", 3: "normaltext"}
    file_name = mapping.get(idx, "normaltext")
    path = os.path.join(BASE_DIR, 'data', f"{file_name}.txt")
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
            return random.choice(lines)
    except:
        return "I'm sorry, I couldn't find my response data."

# --- CHAT LOGIC ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Preprocessing & Prediction
    seq = tokenizer.texts_to_sequences([prompt.lower()])
    padded = pad_sequences(seq, maxlen=20, padding='post')
    pred = model.predict(padded)
    
    # Check if the model is just guessing (bias check)
    if np.max(pred) < 0.3: # If confidence is low, use normaltext
        response = get_response(3)
    else:
        idx = np.argmax(pred)
        response = get_response(idx)

    with st.chat_message("assistant"):
        st.write(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
