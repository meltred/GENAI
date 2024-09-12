import streamlit as st 
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# load the models
model = load_model('next_word_prediction.h5')
## load the tokenizer
tokenizer = pickle.load(open('tokenizer.pickle', 'rb'))

def predict_next_word(seed_text, model, tokenizer, max_sequence_len):
  token_list = tokenizer.texts_to_sequences([seed_text])[0]
  if len(token_list) >= max_sequence_len:
    token_list = token_list[-(max_sequence_len - 1):]
  token_list = pad_sequences([token_list], maxlen=max_sequence_len-1,padding='pre')
  predicted = model.predict(token_list, verbose=0)
  predicted_word = tokenizer.index_word[np.argmax(predicted)]
  return predicted_word


st.title('Next Word Predictor with LSTM')
input_text = st.text_input('Enter a sentence:', 'To be or not to be')
if st.button("Predict Next Word"):
  max_sequence_len = model.input_shape[1] + 1
  predicted_word = predict_next_word(input_text, model, tokenizer, max_sequence_len)
  st.write(f'The next word is: {predicted_word}')
