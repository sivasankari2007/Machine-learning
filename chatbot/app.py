import streamlit as st
import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
import json
import random
import tensorflow as tf
import os

# --- 1. MANDATORY NLTK DOWNLOADS ---
# This prevents the LookupError on Streamlit Cloud
@st.cache_resource
def download_nltk_resources():
    resources = ['punkt', 'punkt_tab', 'wordnet', 'omw-1.4']
    for res in resources:
        nltk.download(res)

download_nltk_resources()

# --- 2. LOAD MODEL AND DATA ---
lemmatizer = WordNetLemmatizer()

# Load intents file
if os.path.exists('chatbot/intents.json'):
    with open('chatbot/intents.json', 'r') as f:
        intents = json.load(f)
else:
    st.error("Error: 'intents.json' not found. Please upload it to your repository.")
    st.stop()

# Load trained model
if os.path.exists('chatbot/chatbot_model.h5'):
    model = tf.keras.models.load_model('chatbot/chatbot_model.h5')
else:
    st.error("Error: 'chatbot_model.h5' not found. Run your training script first.")
    st.stop()

# Recreate words and classes lists (Exactly as done in train.py)
words = []
classes = []
ignore_letters = ['!', '?', ',', '.']

for intent in intents['intents']:
    if intent['tag'] not in classes:
        classes.append(intent['tag'])
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)

words = sorted(list(set([lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_letters])))
classes = sorted(list(set(classes)))

# --- 3. HELPER FUNCTIONS ---
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    return [lemmatizer.lemmatize(word.lower()) for word in sentence_words]

def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def get_response(user_input):
    p = bow(user_input, words)
    res = model.predict(np.array([p]), verbose=0)[0] # verbose=0 hides logs
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    
    if len(results) > 0:
        tag = classes[results[0][0]]
        for i in intents['intents']:
            if i['tag'] == tag:
                return random.choice(i['responses'])
    
    return "I'm sorry, I don't quite understand that."

# --- 4. STREAMLIT UI ---
st.set_page_config(page_title="RNN Chatbot", page_icon="🤖")
st.title("🤖 AI Chatbot")
st.caption("Powered by RNN (LSTM) and Streamlit")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Say something..."):
    # Show user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate and show bot response
    response = get_response(prompt)
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
