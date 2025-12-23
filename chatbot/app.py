import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np
import random

st.set_page_config(page_title="Multi-Intent Chatbot", page_icon="🤖")

# Load model and tokenizer
@st.cache_resource
def load_model_assets():
    model = tf.keras.models.load_model('model/chatbot_rnn_model.h5')
    with open('model/tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_model_assets()

# Function to get a random line from your text files as a response
def get_random_response(category):
    try:
        with open(f"chatbot/data/greetings.txt", 'r', encoding='utf-8') as f:
            lines = f.readlines()
            return random.choice(lines).strip()
    except:
        return "I'm not sure how to respond to that yet."

# UI Logic
st.title("🤖 RNN Smart Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Say hello, ask for a joke or motivation!"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Predict Intent
    seq = tokenizer.texts_to_sequences([prompt])
    padded = pad_sequences(seq, maxlen=20, padding='post')
    pred = model.predict(padded)
    class_idx = np.argmax(pred)

    # Map index to filename
    mapping = {0: "greetings", 1: "joke", 2: "motivation", 3: "normaltext"}
    intent = mapping.get(class_idx)
    
    # Get response from the corresponding text file
    response = get_random_response(intent)

    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
