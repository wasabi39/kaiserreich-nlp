import html
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import Dataset
from transformers import pipeline

# Initialize sentiment analysis pipeline (GPU accelerated)
SENTIMENT_MODEL = "nlptown/bert-base-multilingual-uncased-sentiment"
sentiment_pipeline = pipeline("sentiment-analysis", model=SENTIMENT_MODEL, device=0)

def read_tsv(file_path):
    return pd.read_csv(file_path, delimiter="\t")

def analyze_sentiment(df):
    """Returns sentiment score of each sentences. Using datasets speeds up calculations. """
    dataset = Dataset.from_pandas(df)
    sentiments = dataset.map(lambda x: {"Sentiment": sentiment_pipeline(x['Hit'])[0]['label']})
    df['Sentiment'] = sentiments['Sentiment']
    df["Sentiment"] = df["Sentiment"].str.extract(r"(\d)").astype(int)
    df = df.sort_values(by=['Sentiment', 'Date'])
    return df

def remove_short_and_long_sentences(df, min_length=20, max_length=512):
    #Removes sentences in Hit column longer than max, needed for sentiment analysis
    df['Text_Length'] = df['Hit'].apply(lambda x: len(x))
    df_cleaned = df[df['Text_Length'] >= min_length]
    df_cleaned = df_cleaned[df_cleaned['Text_Length'] <= max_length].drop(columns=['Text_Length'])
    #Remove sentences containing "..."
    df_cleaned = df_cleaned[~df_cleaned["Hit"].str.contains(r"\.\.\.", na=False)]
    return df_cleaned

def remove_duplicates(df):
    initial_count = len(df)
    df_cleaned = df.drop_duplicates(subset='Hit', keep='first')
    removed_duplicates = initial_count - len(df_cleaned)
    print(f"Antal fjernede duplikater: {removed_duplicates}")
    return df_cleaned

def sample_df(df, line_count):
    if len(df) > line_count:
        return df.sample(n=line_count, random_state=42)
    return df

def save_dataframe(df, filename, sentence_count):
    """Gemmer en DataFrame som en CSV-fil i undermappen 'datasets_with_sentiments'."""
    #Hit-column contains a masked token instead of the keyword. We dont want that when saving the file.
    df["Hit"] = df["Unmasked_Hit"]
    df = df.drop(columns=["Unmasked_Hit"])
    
    folder_path = 'datasets_with_sentiments'
    os.makedirs(folder_path, exist_ok=True)
    file_to_save = os.path.join(folder_path, f"{filename}_{sentence_count}.csv")
    df.to_csv(file_to_save, index=False)
    print(f"DataFrame gemt som {file_to_save}")

def shorten_and_clean_df(df, filename, max_length=1000):
    #Shortens df to max_length, used for testing sentiment analysis works on smaller sample
    df_shortened = sample_df(df, max_length)
    df_shortened['Hit'] = df_shortened['Hit'].apply(clean_text)
    #replaces all occurrences of the filename in the Hit column with a masked token
    df_shortened["Unmasked_Hit"] = df_shortened["Hit"]
    pattern = rf"\b{re.escape(filename)}\w*"
    df_shortened["Hit"] = df_shortened["Hit"].str.replace(pattern, "鬯", flags=re.IGNORECASE, regex=True)
    return df_shortened

def clean_text(text):
    return html.unescape(text)

def save_raw_text(df, term, count):
    """Save the hit column to a txt file in the folder 'raw_text'.""" # oversæt til engelsk
    folder_path = 'raw_text'
    os.makedirs(folder_path, exist_ok=True)
    file_to_save = os.path.join(folder_path, f"{term}_{count}.txt")

    with open(file_to_save, 'w', encoding='utf-8') as file:
        for sentence in df['Hit']:
            file.write(sentence + '\n')
    print(f"Raw text gemt som {file_to_save}")

def save_filtered_text(df, term, count):
    """Save the hit column to a txt file in the folder 'raw_text' - only from the correct time period."""
    folder_path = 'filtered_text'
    os.makedirs(folder_path, exist_ok=True)
    file_to_save = os.path.join(folder_path, f"{term}_{count}.txt")
    df_filtered = df.loc[(df['Date'] >= 1871) & (df['Date'] <= 1918), 'Hit']

    with open(file_to_save, 'w', encoding='utf-8') as file:
        for sentence in df_filtered:
            file.write(sentence + '\n')
    
    print(f"Raw text gemt som {file_to_save}")

def plot_stacked_area_chart(df, filename, sentence_count, output_dir="output_stacked_area_chart"):
    os.makedirs(output_dir, exist_ok=True)

    df["Decade"] = (df["Date"] // 1) * 1
    df_counts = df.groupby(["Decade", "Sentiment"]).size().unstack(fill_value=0)

    #Convert counts to proportions
    df_props = df_counts.div(df_counts.sum(axis=1), axis=0)
    df_rolling = df_props.rolling(window=5, center=True, min_periods=1).mean()

    df_rolling.plot(kind="area", stacked=True, colormap="viridis", figsize=(10,6))
    plt.title(f"{filename.capitalize()}: Anteile jeder Sentiment-Bewertung im Zeitverlauf (Gleitender 5-Jahres-Durchschnitt)")
    plt.xlabel("Jahr")
    plt.ylabel("Anteil")
    plt.legend(title="Sentiment")
    plt.grid(True)
    plt.savefig(f"{output_dir}/{filename}_{sentence_count}_stacked_sentiment_plot.png")
    plt.show()

def save_results_to_txt(filename, sentiment_counts):
    output_dir = 'output_results'
    os.makedirs(output_dir, exist_ok=True)

    total_sentiment = 0
    for sentiment, count in sentiment_counts.items():
        total_sentiment += count

    with open(os.path.join(output_dir, f"{filename}_sentiment_counts.txt"), 'w') as f:
        f.write(f"{filename}\nTotal Sentences: {total_sentiment}\n")
        for sentiment in sorted(sentiment_counts.keys()):
            count = sentiment_counts[sentiment]
            f.write(f"{sentiment}: ({count}, {round(count / total_sentiment, 3)})\n")

def run_script():
    filename = "gesund"
    sentence_count_to_use = 1000

    df = read_tsv("Code/" + filename + ".tsv")
    df = shorten_and_clean_df(df, filename, sentence_count_to_use)
    df = remove_short_and_long_sentences(df)
    df = remove_duplicates(df)
    save_raw_text(df, filename, sentence_count_to_use)
    save_filtered_text(df, filename, sentence_count_to_use)
    print("Finished preprocessing")
    df = analyze_sentiment(df)
    print(df.head())
    sentiment_counts = df['Sentiment'].value_counts()
    print(sentiment_counts)
    df_filtered = df[(df['Date'] >= 1871) & (df['Date'] <= 1918)]
    save_dataframe(df, filename, sentence_count_to_use)
    save_results_to_txt(filename, sentiment_counts)
    plot_stacked_area_chart(df_filtered, filename=filename, sentence_count=sentence_count_to_use)

if __name__ == "__main__":
    run_script()
