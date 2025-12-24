import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input

# --- Page Config ---
st.set_page_config(page_title="AI Chatbot", page_icon="🤖")
st.title("🤖 Chatbot")
st.markdown("Type a message below to talk to the bot!")

# --- Load Model & Tokenizer ---
@st.cache_resource
def load_chat_model():
    # Load the main trained model
    model = load_model('model/chatbot_model.h5')
    
    # Load the tokenizer
    with open('model/tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    
    # Reconstruct Inference Models (Encoder & Decoder)
    latent_dim = 256
    
    # Encoder Inference
    enc_inputs = model.input[0]  # input_1
    _, state_h_enc, state_c_enc = model.layers[4].output  # lstm layer
    encoder_model = Model(enc_inputs, [state_h_enc, state_c_enc])
    
    # Decoder Inference
    dec_inputs = model.input[1]  # input_2
    dec_state_input_h = Input(shape=(latent_dim,))
    dec_state_input_c = Input(shape=(latent_dim,))
    dec_states_inputs = [dec_state_input_h, dec_state_input_c]
    
    dec_lstm = model.layers[5]
    dec_outputs, state_h_dec, state_c_dec = dec_lstm(
        model.layers[3](dec_inputs), initial_state=dec_states_inputs
    )
    dec_dense = model.layers[6]
    dec_outputs = dec_dense(dec_outputs)
    
    decoder_model = Model(
        [dec_inputs] + dec_states_inputs, 
        [dec_outputs] + [state_h_dec, state_c_dec]
    )
    
    return tokenizer, encoder_model, decoder_model

tokenizer, encoder_model, decoder_model = load_chat_model()
max_len = 20

# --- Response Generation Logic ---
def get_bot_response(user_input):
    input_seq = tokenizer.texts_to_sequences([user_input.lower()])
    input_seq = pad_sequences(input_seq, maxlen=max_len, padding='post')
    
    states_value = encoder_model.predict(input_seq, verbose=0)
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = tokenizer.word_index.get('startseq', 1)
    
    stop_condition = False
    decoded_sentence = ""
    
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value, verbose=0)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = tokenizer.index_word.get(sampled_token_index, '')
        
        if sampled_word == 'endseq' or len(decoded_sentence.split()) > max_len:
            stop_condition = True
        else:
            decoded_sentence += " " + sampled_word
            
        target_seq[0, 0] = sampled_token_index
        states_value = [h, c]
        
    return decoded_sentence.strip()

# --- Chat UI ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Say something..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate bot response
    with st.chat_message("assistant"):
        response = get_bot_response(prompt)
        if not response:
            response = "I'm not sure how to respond to that."
        st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
