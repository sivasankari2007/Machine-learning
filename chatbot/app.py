import streamlit as st
import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
import json
import random
import tensorflow as tf
import os
from datetime import datetime
import webbrowser # Tool for opening links

# --- 1. SETUP & DOWNLOADS ---
@st.cache_resource
def download_resources():
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('wordnet')

download_resources()
lemmatizer = WordNetLemmatizer()

# --- 2. LOAD DATA ---
model = tf.keras.models.load_model('chatbot/chatbot_model.h5')
with open('intents.json', 'r') as f:
    intents = json.load(f)

# Recreate vocabulary from intents.json
words = []
classes = []
for intent in intents['intents']:
    if intent['tag'] not in classes: classes.append(intent['tag'])
    for pattern in intent['patterns']:
        words.extend(nltk.word_tokenize(pattern))
words = sorted(list(set([lemmatizer.lemmatize(w.lower()) for w in words if w not in ['?', '!', '.', ',']])))
classes = sorted(list(set(classes)))

# --- 3. THE TOOLS (NEW FEATURE) ---
def get_system_time():
    return datetime.now().strftime("%I:%M %p")

def search_google(query):
    return f"https://www.google.com/search?q={query.replace(' ', '+')}"

# --- 4. PROCESSING LOGIC ---
def get_response(user_input):
    # Convert input to Bag of Words
    sentence_words = nltk.word_tokenize(user_input)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    bag = [1 if w in sentence_words else 0 for w in words]
    
    # Predict
    res = model.predict(np.array([bag]), verbose=0)[0]
    idx = np.argmax(res)
    tag = classes[idx]
    prob = res[idx]

    if prob > 0.5:
        # Check for Tool Triggers
        if tag == "time":
            return f"The current time is {get_system_time()}"
        if tag == "google":
            return f"I've prepared a search for you: [Click here]({search_google(user_input)})"
        
        # Default response from JSON
        for i in intents['intents']:
            if i['tag'] == tag:
                return random.choice(i['responses'])
    
    return "I'm not sure I understand. Can you try again?"

# --- 5. STREAMLIT UI ---
st.set_page_config(page_title="Smart RNN Assistant", page_icon="🧠")

# Sidebar Tools
with st.sidebar:
    st.header("🛠️ Internal Tools")
    st.write(f"📅 Date: {datetime.now().strftime('%Y-%m-%d')}")
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

st.title("🧠 Smart RNN Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask me a joke or the time..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    response = get_response(prompt)
    
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
