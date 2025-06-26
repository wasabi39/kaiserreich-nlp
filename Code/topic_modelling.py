import os
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pyLDAvis

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RANDOM_SEED = 42
MIN_TEXTS_REQUIRED = 20
DEFAULT_TOPICS = 3
DEFAULT_WORDS = 30

nltk.download('stopwords', quiet=True)

CUSTOM_STOPWORDS = [
    "mitteleuropa", "mitteleuropas", "mitteleuropäisch", "mitteleuropäische", 
    "mitteleuropäischen", "mitteleuropäischem", "mitteleuropäischer", "mitteleuropäisches"
]

def get_code_path(relative_path):
    return os.path.join(SCRIPT_DIR, relative_path)

def load_sentiment_data():
    df1 = pd.read_csv(get_code_path("datasets_with_sentiments/mitteleuropa_100000.csv"))
    df2 = pd.read_csv(get_code_path("datasets_with_sentiments/mitteleuropäisch_100000.csv"))
    
    combined_df = pd.concat([df1, df2], ignore_index=True)
    return combined_df[["Hit", "Sentiment"]].dropna()

def categorize_sentiment(score):
    if score in [1, 2]:
        return "negativ"
    elif score == 3:
        return "neutral"
    else:
        return "positiv"

def group_texts_by_sentiment(df):
    df["gruppe"] = df["Sentiment"].apply(categorize_sentiment)
    
    groups = {}
    for sentiment in ["negativ", "neutral", "positiv"]:
        groups[sentiment] = df[df["gruppe"] == sentiment]["Hit"].tolist()
    
    return groups

def create_lda_model(texts, n_topics=DEFAULT_TOPICS):
    german_stopwords = stopwords.words('german') + CUSTOM_STOPWORDS
    
    vectorizer = CountVectorizer(
        stop_words=german_stopwords, 
        max_df=0.95, 
        min_df=2
    )
    document_matrix = vectorizer.fit_transform(texts)
    
    lda_model = LatentDirichletAllocation(
        n_components=n_topics, 
        random_state=RANDOM_SEED
    )
    lda_model.fit(document_matrix)
    
    return lda_model, vectorizer, document_matrix

def print_topics(lda_model, vectorizer, group_name, text_count, n_words=DEFAULT_WORDS):
    words = vectorizer.get_feature_names_out()
    
    print(f"\nLDA Topics for {group_name.upper()} ({text_count} texts):")
    
    for topic_idx, topic in enumerate(lda_model.components_):
        top_word_indices = topic.argsort()[:-n_words - 1:-1]
        top_words = [words[idx] for idx in top_word_indices]
        print(f"Topic {topic_idx + 1}: {', '.join(top_words)}")

def save_visualization(lda_model, vectorizer, document_matrix, group_name):
    words = vectorizer.get_feature_names_out()
    
    doc_topic_distributions = lda_model.transform(document_matrix)
    topic_term_distributions = lda_model.components_ / lda_model.components_.sum(axis=1)[:, None]
    
    doc_lengths = document_matrix.sum(axis=1).A1
    vocabulary = words.tolist()
    term_frequencies = document_matrix.sum(axis=0).A1
    
    visualization_data = pyLDAvis.prepare(
        topic_term_dists=topic_term_distributions,
        doc_topic_dists=doc_topic_distributions,
        doc_lengths=doc_lengths,
        vocab=vocabulary,
        term_frequency=term_frequencies
    )
    
    output_dir = get_code_path("lda_results")
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, f"lda_vis_{group_name}.html")
    pyLDAvis.save_html(visualization_data, output_file)
    
    print(f"Visualization saved to: {output_file}")

def analyze_sentiment_group(texts, group_name, n_topics=DEFAULT_TOPICS, n_words=DEFAULT_WORDS):
    if len(texts) < MIN_TEXTS_REQUIRED:
        print(f"Insufficient texts in group '{group_name}' (need at least {MIN_TEXTS_REQUIRED})")
        return
    
    lda_model, vectorizer, document_matrix = create_lda_model(texts, n_topics)
    print_topics(lda_model, vectorizer, group_name, len(texts), n_words)
    save_visualization(lda_model, vectorizer, document_matrix, group_name)

def main():
    df = load_sentiment_data()
    sentiment_groups = group_texts_by_sentiment(df)
    
    for group_name, texts in sentiment_groups.items():
        analyze_sentiment_group(texts, group_name)

if __name__ == "__main__":
    main()
