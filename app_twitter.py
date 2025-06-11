import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud
from nltk.corpus import stopwords

st.title('Analisis Sentimen Data X/Twitter: AI-replacing-jobs')

# Sidebar untuk upload file
uploaded_file = st.sidebar.file_uploader('Upload file CSV Twitter', type=['csv'])

if uploaded_file:
    file_name = uploaded_file
else:
    file_name = 'AI-replacing-jobs.csv'  # fallback default

# Download resource NLTK
try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Load data
def load_data(file):
    try:
        df = pd.read_csv(file)
        return df
    except Exception as e:
        st.error(f'Gagal memuat data: {e}')
        return None

df = load_data(file_name)

if df is not None:
    st.subheader('Eksplorasi Data Awal')
    st.write('5 Baris Pertama Data:')
    st.dataframe(df.head())
    st.write('Info DataFrame:')
    buffer = []
    df.info(buf=buffer)
    st.text('\n'.join(buffer))
    st.write('Statistik Deskriptif:')
    st.dataframe(df.describe())
    st.write('Jumlah Nilai Hilang per Kolom:')
    st.write(df.isnull().sum())

    # Data cleaning
    if 'full_text' in df.columns:
        df.dropna(subset=['full_text'], inplace=True)
        df['content'] = df['full_text'].fillna('')
        df.drop(columns=['full_text'], inplace=True)
    else:
        st.error("Kolom 'full_text' tidak ditemukan pada data.")

    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'@\S+|#\S+', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    df['cleaned_content'] = df['content'].apply(clean_text)

    st.subheader('Contoh Kolom content dan cleaned_content')
    st.dataframe(df[['content', 'cleaned_content']].head())

    # Sentiment analysis
    analyzer = SentimentIntensityAnalyzer()
    def get_vader_sentiment(text):
        if not text:
            return {'neg': 0.0, 'neu': 0.0, 'pos': 0.0, 'compound': 0.0}
        return analyzer.polarity_scores(text)
    sentiment_scores = df['cleaned_content'].apply(get_vader_sentiment)
    df['sentiment_neg'] = sentiment_scores.apply(lambda x: x['neg'])
    df['sentiment_neu'] = sentiment_scores.apply(lambda x: x['neu'])
    df['sentiment_pos'] = sentiment_scores.apply(lambda x: x['pos'])
    df['sentiment_compound'] = sentiment_scores.apply(lambda x: x['compound'])

    def categorize_sentiment(compound_score):
        if compound_score >= 0.05:
            return 'Positive'
        elif compound_score <= -0.05:
            return 'Negative'
        else:
            return 'Neutral'
    df['sentiment_label'] = df['sentiment_compound'].apply(categorize_sentiment)

    st.subheader('Contoh Data dengan Skor Sentimen dan Label')
    st.dataframe(df[['cleaned_content', 'sentiment_neg', 'sentiment_neu', 'sentiment_pos', 'sentiment_compound', 'sentiment_label']].head())
    st.write('Distribusi Label Sentimen:')
    st.write(df['sentiment_label'].value_counts())

    # Visualisasi distribusi sentimen
    st.subheader('Visualisasi Distribusi Sentimen')
    distribusi = df['sentiment_label'].value_counts().reindex(['Negative', 'Neutral', 'Positive'])
    fig, ax = plt.subplots(figsize=(7,5))
    sns.barplot(x=distribusi.index, y=distribusi.values, palette='viridis', ax=ax)
    ax.set_title('Distribusi Label Sentimen pada Data X/Twitter', fontsize=15)
    ax.set_xlabel('Label Sentimen', fontsize=12)
    ax.set_ylabel('Jumlah Data', fontsize=12)
    for i, v in enumerate(distribusi.values):
        ax.text(i, v + 2, str(v), ha='center', va='bottom', fontsize=11)
    st.pyplot(fig)

    # Word Cloud
    st.subheader('Word Cloud per Sentimen')
    my_stopwords = set(stopwords.words('english'))
    additional_stopwords = ['ai', 'work', 'human', 'people', 'can', 'will', 'like', 'just', 'get', 'one', 'new', 'would', 'also', 'dont', 'think', 'many', 'much', 'even', 'could', 'use', 'know', 'way', 'time', 'etc', 'make', 'really', 'see', 'good']
    my_stopwords.update(additional_stopwords)
    def generate_wordcloud(text_data, title):
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            stopwords=my_stopwords,
            min_font_size=10
        ).generate(text_data)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.set_title(title, fontsize=18)
        ax.axis('off')
        st.pyplot(fig)
    st.write('Word Cloud - Sentimen Positif')
    positive_text = ' '.join(df[df['sentiment_label'] == 'Positive']['cleaned_content'])
    generate_wordcloud(positive_text, 'Word Cloud - Sentimen Positif')
    st.write('Word Cloud - Sentimen Negatif')
    negative_text = ' '.join(df[df['sentiment_label'] == 'Negative']['cleaned_content'])
    generate_wordcloud(negative_text, 'Word Cloud - Sentimen Negatif')
    st.write('Word Cloud - Sentimen Netral')
    neutral_text = ' '.join(df[df['sentiment_label'] == 'Neutral']['cleaned_content'])
    generate_wordcloud(neutral_text, 'Word Cloud - Sentimen Netral')
else:
    st.warning('Data belum dimuat. Silakan upload file CSV yang sesuai.')
