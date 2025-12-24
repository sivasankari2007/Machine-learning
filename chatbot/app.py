import streamlit as st
import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
import json
import random
import tensorflow as tf

# Load resources
lemmatizer = WordNetLemmatizer()
model = tf.keras.models.load_model('chatbot/chatbot_model.h5')
intents = json.loads(open('intents.json').read())

# These must match exactly what you used in your train.py
# If you didn't save them as pkl, ensure you generate them from intents.json here
words = sorted(list(set([lemmatizer.lemmatize(w.lower()) for intent in intents['intents'] 
                        for pattern in intent['patterns'] 
                        for w in nltk.word_tokenize(pattern) if w not in ['!', '?', ',', '.']])))
classes = sorted(list(set([intent['tag'] for intent in intents['intents']])))

# Helper functions
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    return [lemmatizer.lemmatize(word.lower()) for word in sentence_words]

def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s: bag[i] = 1
    return np.array(bag)

def get_response(user_input):
    p = bow(user_input, words)
    res = model.predict(np.array([p]))[0]
    results = [[i, r] for i, r in enumerate(res) if r > 0.25]
    results.sort(key=lambda x: x[1], reverse=True)
    
    if results:
        tag = classes[results[0][0]]
        for i in intents['intents']:
            if i['tag'] == tag:
                return random.choice(i['responses'])
    return "I'm sorry, I don't understand that."

# --- STREAMLIT UI ---
st.title("🤖 RNN Chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("How can I help you?"):
    # Display user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get bot response
    response = get_response(prompt)
    
    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
