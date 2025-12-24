import streamlit as st
import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
import json
import random
import tensorflow as tf
import os
from datetime import datetime
import urllib.parse

# --- 1. INITIAL SETUP ---
@st.cache_resource
def download_nltk_data():
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('wordnet')

download_nltk_data()
lemmatizer = WordNetLemmatizer()

# --- 2. LOAD AI RESOURCES ---
@st.cache_resource
def load_chatbot_files():
    # Verify files exist before loading
    files = ['chatbot_model.h5', 'intents.json', 'data.json']
    for f in files:
        if not os.path.exists(f):
            st.error(f"⚠️ Critical file missing: {f}. Please run train.py and upload it.")
            st.stop()

    model = tf.keras.models.load_model('chatbot_model.h5')
    
    with open('intents.json', 'r') as f:
        intents = json.load(f)
        
    with open('data.json', 'r') as f:
        data = json.load(f)
        words = data['words']
        classes = data['classes']
        
    return model, intents, words, classes

model, intents, words, classes = load_chatbot_files()

# --- 3. CORE LOGIC ---
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    return [lemmatizer.lemmatize(w.lower()) for w in sentence_words]

def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    # The fix for ValueError: Ensure bag is EXACTLY the same length as training words
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s: bag[i] = 1
    return np.array(bag)

def get_response(user_input):
    p = bow(user_input, words)
    # Predicting with the model
    res = model.predict(np.array([p]), verbose=0)[0]
    results_index = np.argmax(res)
    tag = classes[results_index]
    confidence = res[results_index]

    if confidence > 0.5:
        # TOOL: Current Time
        if tag == "time":
            return f"🕒 The current time is {datetime.now().strftime('%H:%M:%S')}", None
        
        # TOOL: Image Generation (Pollinations API)
        if tag == "generate_image":
            prompt = user_input.lower()
            for trigger in ["draw a", "generate an image of", "create a picture of"]:
                prompt = prompt.replace(trigger, "")
            img_url = f"https://image.pollinations.ai/prompt/{urllib.parse.quote(prompt.strip())}?nologo=true"
            return "🎨 Here is what I created for you:", img_url

        # TOOL: Jokes / General QA
        for i in intents['intents']:
            if i['tag'] == tag:
                return random.choice(i['responses']), None

    return "I'm not quite sure. Try asking for a joke or to draw something!", None

# --- 4. STREAMLIT UI ---
st.set_page_config(page_title="AI Multi-Bot", page_icon="🤖")
st.title("🤖 RNN Multi-Tool Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Show chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("image"):
            st.image(message["image"])

# Input handling
if prompt := st.chat_input("Say hi, ask for a joke, or say 'Draw a space cat'"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    response_text, image_url = get_response(prompt)

    with st.chat_message("assistant"):
        st.markdown(response_text)
        if image_url:
            st.image(image_url)

    st.session_state.messages.append({
        "role": "assistant", 
        "content": response_text, 
        "image": image_url
    })
