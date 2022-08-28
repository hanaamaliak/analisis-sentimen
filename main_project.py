import streamlit as st
import re
import string
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from tensorflow.keras import models, utils
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import plotly.express as px
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from wordcloud import WordCloud, STOPWORDS


st.title("Analisis Sentimen Publik di Twitter terhadap Rencana Pemindahan Ibu Kota Negara (IKN)")
st.markdown("### By: Hana Amalia Kushandini")
st.write("")
st.sidebar.title("Pilihan")

st.sidebar.markdown("Berikut merupakan beberapa hal yang bisa dilihat di website ini")

data_path = ("https://raw.githubusercontent.com/varaah/analisis-sentimen-ikn/main/data/twitterIKN-labelled_new.csv")

@st.cache(persist=True)
def load_data():
    data = pd.read_csv(data_path)
    data['sentiment'] = np.where(data['target']==0, 'Negatif', 'Positif')
    return data

data = load_data()

#tampilan utama (word cloud)
word_sentiment = st.radio("Pilih Sentimen: ", tuple(pd.unique(data["sentiment"])))
st.markdown("##### Word Cloud untuk Sentimen " + word_sentiment)
with st.spinner("Loading..."):
    df = data[data["sentiment"]==word_sentiment]
    words = " ".join(df["tweet"])
    processed_words = " ".join([word for word in words.split() if "http" not in word and not word.startswith("@") and word != "RT"])
    wordcloud = WordCloud(stopwords=STOPWORDS, background_color="white", width=800, height=640).generate(processed_words)
    st.set_option('deprecation.showPyplotGlobalUse', False)   
    plt.imshow(wordcloud)
    plt.xticks([])
    plt.yticks([])
    st.pyplot()

#lihat random tweet
st.sidebar.subheader("Lihat Tweet Random")
random_tweet = st.sidebar.selectbox('Pilih sentimen',('Positif','Negatif'))
if st.sidebar.checkbox('Tampilkan',False):
    st.write('-----------------------------------------------------------------------------------------------------------')
    st.markdown("##### Contoh Tweet dengan Sentimen " + random_tweet)
    st.markdown("1." + data.query("sentiment == @random_tweet")[['tweet']].sample(n=1).iat[0,0])

#prediksi sentimen
st.sidebar.subheader("Prediksi Sentimen")
if st.sidebar.checkbox('Mulai Prediksi', False):
    st.write('-----------------------------------------------------------------------------------------------------------')
    st.markdown("##### Memprediksi Sentimen Pendapat Kamu")
    pred_review_text = st.text_area("Masukkan kalimat:")
    if st.button('Prediksi'):
        with st.spinner("Loading..."):
            model = load_model("model_fix.h5")
        if pred_review_text != '':
            pred = []
            pred.append(pred_review_text)
            with st.spinner("Loading..."):
                pred1 = list(map(str.lower,pred))
                pred2 = [word.translate(string.punctuation) for word in pred1]
                factory = StemmerFactory()
                stemmer = factory.create_stemmer()
                pred3 = [" ".join([stemmer.stem(word) for word in sentence.split(" ")]) for sentence in pred2]
                tokenizer = keras.preprocessing.text.Tokenizer(num_words = 100000)
                tokenizer.fit_on_texts(pred3)
                pred_seq = tokenizer.texts_to_sequences(pred3)
                pred_padded = keras.preprocessing.sequence.pad_sequences(pred_seq, maxlen = 48, padding='post')
            val = model.predict(pred_padded)
            predictions = np.where(val > 0.5, "Positif", "Negatif")
            predictions = str(predictions)
            rep = {"]]": "", "[[": ""}
            rep = dict((re.escape(k), v) for k, v in rep.items()) 
            pattern = re.compile("|".join(rep.keys()))
            predictions = pattern.sub(lambda m: rep[re.escape(m.group(0))], predictions)
            st.write('Prediksi Sentimen: ' + predictions)

#lihat visualisasi
st.sidebar.subheader("Visualisasi Tweet per Sentimen")
select = st.sidebar.selectbox('Tipe Visualisasi', ['Histogram','PieChart'])

sentiment_count = data['sentiment'].value_counts()
sentiment_count = pd.DataFrame({'Sentiments':sentiment_count.index,'Tweets':sentiment_count.values})

if st.sidebar.checkbox('Tampilkan',False,key='0'):
    st.write('-----------------------------------------------------------------------------------------------------------')
    st.markdown("#### Banyaknya Tweet per Sentimen")
    if select=='Histogram':
        fig = px.bar(sentiment_count, x='Sentiments', y='Tweets', color='Sentiments',height=500)
        st.plotly_chart(fig)
    else:
        fig = px.pie(sentiment_count,values='Tweets',names='Sentiments')
        st.plotly_chart(fig)
