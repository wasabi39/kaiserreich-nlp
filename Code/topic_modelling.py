import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

# Download stopwords if not already present
nltk.download('stopwords')
german_stop_words = stopwords.words('german') + ["mitteleuropa", "mitteleuropas", "mitteleuropäisch", "mitteleuropäische", "mitteleuropäischen", "mitteleuropäischem", "mitteleuropäischer", "mitteleuropäisches"]

# ----- 1. Indlæs data -----
df1 = pd.read_csv("datasets_with_sentiments/mitteleuropa_100000.csv")
df2 = pd.read_csv("datasets_with_sentiments/mitteleuropäisch_100000.csv")

# Kombinér dem til én DataFrame
df = pd.concat([df1, df2], ignore_index=True)
df = df[["Hit", "Sentiment"]].dropna()

# ----- 2. Del i grupper -----
def get_group(score):
    if score in [1, 2]:
        return "negativ"
    elif score == 3:
        return "neutral"
    else:
        return "positiv"

df["gruppe"] = df["Sentiment"].apply(get_group)

groups = {
    "negativ": df[df["gruppe"] == "negativ"]["Hit"].tolist(),
    "neutral": df[df["gruppe"] == "neutral"]["Hit"].tolist(),
    "positiv": df[df["gruppe"] == "positiv"]["Hit"].tolist()
}

# ----- 3. Funktion til topic modelling med LDA -----
def show_topics_lda(texts, gruppenavn, n_topics=3, n_words=30):
    if len(texts) < 20:
        print(f"⚠️ Ikke nok tekster i gruppe '{gruppenavn}'")
        return
    
    vectorizer = CountVectorizer(stop_words=german_stop_words, max_df=0.95, min_df=2)
    X = vectorizer.fit_transform(texts)

    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(X)
    
    words = vectorizer.get_feature_names_out()

    print(f"\n🧠 LDA Topics for gruppe: {gruppenavn.upper()} ({len(texts)} tekster):\n")
    for i, topic in enumerate(lda.components_):
        top_words = [words[j] for j in topic.argsort()[:-n_words - 1:-1]]
        print(f"🔹 Topic {i+1}: {', '.join(top_words)}")

    # Prepare the components needed for pyLDAvis
    doc_topic_dists = lda.transform(X)  # shape: n_docs x n_topics
    topic_term_dists = lda.components_ / lda.components_.sum(axis=1)[:, None]  # normalize to probs

    doc_lengths = X.sum(axis=1).A1  # total words per document, flatten to 1D array
    vocab = words.tolist()
    term_frequency = X.sum(axis=0).A1  # frequency of each term in corpus

    data = pyLDAvis.prepare(
        topic_term_dists=topic_term_dists,
        doc_topic_dists=doc_topic_dists,
        doc_lengths=doc_lengths,
        vocab=vocab,
        term_frequency=term_frequency
    )

    pyLDAvis.save_html(data, f"lda_vis_{gruppe}.html")

# ----- 4. Kør for alle grupper -----
for gruppe, tekster in groups.items():
    show_topics_lda(tekster, gruppe)
