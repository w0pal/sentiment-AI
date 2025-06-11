import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud
from nltk.corpus import stopwords

st.title('Analisis Sentimen Reddit: AI dan Pekerjaan')

# Sidebar untuk upload file
uploaded_file = st.sidebar.file_uploader('Upload file CSV Reddit', type=['csv'])

if uploaded_file:
    file_name = uploaded_file
else:
    file_name = 'JUMBO-BADAG.csv'  # fallback default

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
    st.write('Jumlah Unik pada Kolom Kunci:')
    st.write({
        'type': df['type'].nunique(),
        'subreddit': df['subreddit'].nunique(),
        'author': df['author'].nunique()
    })
    st.write('Jumlah Nilai Hilang per Kolom:')
    st.write(df.isnull().sum())
    st.write('Distribusi Tipe Data (Post/Comment):')
    st.write(df['type'].value_counts())

    # Data cleaning
    df.dropna(subset=['text'], inplace=True)
    df['content'] = ''
    df.loc[df['type'] == 'post', 'content'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
    df.loc[df['type'] == 'comment', 'content'] = df['text']
    if 'title' in df.columns and 'text' in df.columns:
        df.drop(columns=['title', 'text'], inplace=True)
    elif 'text' in df.columns:
        df.drop(columns=['text'], inplace=True)

    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'u/\S+|r/\S+', '', text)
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
    df['sentiment_scores'] = df['cleaned_content'].apply(get_vader_sentiment)
    df['sentiment_neg'] = df['sentiment_scores'].apply(lambda x: x['neg'])
    df['sentiment_neu'] = df['sentiment_scores'].apply(lambda x: x['neu'])
    df['sentiment_pos'] = df['sentiment_scores'].apply(lambda x: x['pos'])
    df['sentiment_compound'] = df['sentiment_scores'].apply(lambda x: x['compound'])
    df.drop(columns=['sentiment_scores'], inplace=True)

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
    sentiment_counts = df['sentiment_label'].value_counts()
    order = ['Negative', 'Neutral', 'Positive']
    sentiment_counts = sentiment_counts.reindex(order)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette='viridis', ax=ax)
    ax.set_title('Distribusi Sentimen Terhadap AI dan Pekerjaan di Reddit')
    ax.set_xlabel('Label Sentimen')
    ax.set_ylabel('Jumlah Entri')
    for index, value in enumerate(sentiment_counts.values):
        ax.text(index, value + 0.5, str(value), ha='center', va='bottom')
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
