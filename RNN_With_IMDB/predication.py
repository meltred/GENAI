import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense


word_index = imdb.get_word_index()
#word_index

reverse_word_index = {value: key for key, value in word_index.items()}

model = load_model('simple_rnn.h5')

def decode_review(enc_review):
    decode_review = ' '.join([reverse_word_index.get(i - 3, '?') for i  in enc_review])
    return decode_review
def preprocess_text(text):
    words = text.lower().split()
    enc_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([enc_review],maxlen=500)
    return padded_review


##predication function
def predict_sentiment(review):
    preprocessed_input = preprocess_text(review)
    predication =model.predict(preprocessed_input)
    sentiment = 'Positive' if predication[0][0] > 0.5 else 'Negative'

    return sentiment, predication[0][0]

def main():
    inp = input('enter the text')
    print(predict_sentiment(inp))
if __name__ == "__main__":
    main()

