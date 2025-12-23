import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np
import random

# --- PAGE CONFIG & GOOGLE COLORS ---
st.set_page_config(page_title="Google AI Assistant", page_icon="🔍", layout="centered")

# Custom CSS for Google Theme
st.markdown("""
    <style>
    .stApp {
        background-color: #FFFFFF;
    }
    .main-header {
        font-family: 'Product Sans', Arial, sans-serif;
        color: #4285F4;
        font-size: 48px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 0px;
    }
    .google-span { display: inline-block; }
    .g-blue { color: #4285F4; }
    .g-red { color: #EA4335; }
    .g-yellow { color: #FBBC05; }
    .g-green { color: #34A853; }
    
    /* Styling chat bubbles */
    .stChatMessage {
        border-radius: 20px;
        padding: 10px;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- GOOGLE LOGO HEADER ---
st.markdown("""
    <div class='main-header'>
        <span class='g-blue'>G</span><span class='g-red'>o</span><span class='g-yellow'>o</span><span class='g-blue'>g</span><span class='g-green'>l</span><span class='g-red'>e</span>
        <span style='color:#5f6368; font-size: 24px;'>Chatbot</span>
    </div>
    """, unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #5f6368;'>RNN Machine Learning Project</p>", unsafe_allow_html=True)

# --- LOAD ASSETS ---
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model('model/chatbot_rnn_model.h5')
    with open('model/tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

try:
    model, tokenizer = load_assets()
except Exception as e:
    st.error("Model files not found. Please run train.py first!")

def get_bot_response(intent_idx):
    mapping = {0: "greetings", 1: "joke", 2: "motivation", 3: "normaltext"}
    filename = mapping.get(intent_idx)
    try:
        with open(f"data/{filename}.txt", 'r', encoding='utf-8') as f:
            lines = f.readlines()
            return random.choice(lines).strip()
    except:
        return "I'm searching for the best answer for you..."

# --- CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    avatar = "👤" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        st.write(msg["content"])

if prompt := st.chat_input("Search or ask anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.write(prompt)

    # Prediction
    seq = tokenizer.texts_to_sequences([prompt.lower()])
    padded = pad_sequences(seq, maxlen=20, padding='post')
    prediction = model.predict(padded)
    idx = np.argmax(prediction)
    
    response = get_bot_response(idx)

    with st.chat_message("assistant", avatar="🤖"):
        st.info(response) # Blue-ish box for response
    st.session_state.messages.append({"role": "assistant", "content": response})
