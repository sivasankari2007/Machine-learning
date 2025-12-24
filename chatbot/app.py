import streamlit as st
import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
import json
import tensorflow as tf
import os
import urllib.parse

# --- SETUP ---
@st.cache_resource
def load_all_resources():
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('wordnet')
    
    # Load the Model
    model = tf.keras.models.load_model('chatbot/chatbot_model.h5')
    
    # Load Intents
    with open('intents.json', 'r') as f:
        intents = json.load(f)
        
    # Load Vocabulary (The Fix for ValueError)
    with open('data.json', 'r') as f:
        data = json.load(f)
        words = data['words']
        classes = data['classes']
        
    return model, intents, words, classes

model, intents, words, classes = load_all_resources()
lemmatizer = WordNetLemmatizer()

def get_response(user_input):
    # 1. Preprocess input
    sentence_words = nltk.word_tokenize(user_input)
    sentence_words = [lemmatizer.lemmatize(w.lower()) for w in sentence_words]
    
    # 2. Create Bag of Words (Must be same length as 'words')
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    
    # 3. Predict
    # This np.array([bag]) will now match the model's input shape perfectly
    results = model.predict(np.array([bag]), verbose=0)[0]
    idx = np.argmax(results)
    tag = classes[idx]
    
    # 4. Logic for Tools
    if results[idx] > 0.6:
        if tag == "generate_image":
            # Image logic
            clean_prompt = user_input.lower().replace("draw", "").replace("generate", "")
            img_url = f"https://image.pollinations.ai/prompt/{urllib.parse.quote(clean_prompt)}?nologo=true"
            return "🎨 Here is your image:", img_url
        
        for i in intents['intents']:
            if i['tag'] == tag:
                return random.choice(i['responses']), None
                
    return "I'm not sure about that.", None

# --- STREAMLIT UI CODE REMAINS THE SAME ---
