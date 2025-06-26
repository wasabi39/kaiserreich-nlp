import html
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from datasets import Dataset
from transformers import pipeline

FILE_NAME= "gesund"
SENTIMENT_MODEL = "nlptown/bert-base-multilingual-uncased-sentiment"
GERMAN_EMPIRE_START = 1871
GERMAN_EMPIRE_END = 1918
DEFAULT_MIN_LENGTH = 20
DEFAULT_MAX_LENGTH = 512
DEFAULT_SAMPLE_SIZE = 1000
RANDOM_SEED = 42
MASK_TOKEN = "鬯"

# Get the directory where this script is located (Code folder)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def get_code_path(relative_path):
    """Ensure all paths are relative to the Code directory."""
    return os.path.join(SCRIPT_DIR, relative_path)

# Initialize sentiment analysis pipeline (device=0 means we run on GPU, requires a 2.7GB download)
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

def remove_short_and_long_sentences(df, min_length=DEFAULT_MIN_LENGTH, max_length=DEFAULT_MAX_LENGTH):
    """Remove sentences that are too short, too long, or contain ellipsis."""
    df['text_length'] = df['Hit'].str.len()
    df_cleaned = df[(df['text_length'] >= min_length) & (df['text_length'] <= max_length)]
    df_cleaned = df_cleaned.drop(columns=['text_length'])
    # Remove sentences containing "..."
    df_cleaned = df_cleaned[~df_cleaned["Hit"].str.contains(r"\.\.\.", na=False)]
    return df_cleaned

def remove_duplicates(df):
    initial_count = len(df)
    df_cleaned = df.drop_duplicates(subset='Hit', keep='first')
    removed_duplicates = initial_count - len(df_cleaned)
    print(f"Removed duplicates: {removed_duplicates}")
    return df_cleaned

def sample_df(df, line_count):
    """Sample dataframe if it's larger than line_count."""
    if len(df) > line_count:
        return df.sample(n=line_count, random_state=RANDOM_SEED)
    return df

def save_dataframe(df, filename, sentence_count):
    """Save a DataFrame as a CSV file in the 'datasets_with_sentiments' subfolder."""
    #Hit-column contains a masked token instead of the keyword. We dont want that when saving the file.
    df["Hit"] = df["Unmasked_Hit"]
    df = df.drop(columns=["Unmasked_Hit"])
    
    folder_path = get_code_path('datasets_with_sentiments')
    os.makedirs(folder_path, exist_ok=True)
    file_to_save = os.path.join(folder_path, f"{filename}_{sentence_count}.csv")
    df.to_csv(file_to_save, index=False)
    print(f"DataFrame saved as {file_to_save}")

def shorten_and_clean_df(df, filename, max_length=DEFAULT_SAMPLE_SIZE):
    """Shortens df to max_length, used for testing sentiment analysis on smaller sample."""
    df_shortened = sample_df(df, max_length)
    df_shortened['Hit'] = df_shortened['Hit'].apply(clean_text)
    # Replace all occurrences of the filename in the Hit column with a masked token
    df_shortened["Unmasked_Hit"] = df_shortened["Hit"]
    pattern = rf"\b{re.escape(filename)}\w*"
    df_shortened["Hit"] = df_shortened["Hit"].str.replace(pattern, MASK_TOKEN, flags=re.IGNORECASE, regex=True)
    return df_shortened

def clean_text(text):
    return html.unescape(text)

def save_text_to_file(data, folder_path, filename):
    full_folder_path = get_code_path(folder_path)
    os.makedirs(full_folder_path, exist_ok=True)
    file_path = os.path.join(full_folder_path, filename)
    
    with open(file_path, 'w', encoding='utf-8') as file:
        for item in data:
            file.write(f"{item}\n")
    
    print(f"Text saved as {file_path}")
    return file_path

def save_raw_text(df, term, count):
    return save_text_to_file(df['Hit'], 'raw_text', f"{term}_{count}.txt")

def save_filtered_text(df, term, count):
    filtered_data = df.loc[(df['Date'] >= GERMAN_EMPIRE_START) & (df['Date'] <= GERMAN_EMPIRE_END), 'Hit']
    return save_text_to_file(filtered_data, 'filtered_text', f"{term}_{count}.txt")

def plot_stacked_area_chart(df, filename, sentence_count, output_dir="output_stacked_area_chart"):
    """Create a stacked area chart showing sentiment distribution over time."""
    full_output_dir = get_code_path(output_dir)
    os.makedirs(full_output_dir, exist_ok=True)
    df_counts = df.groupby(["Date", "Sentiment"]).size().unstack(fill_value=0)

    #Convert counts to proportions
    df_props = df_counts.div(df_counts.sum(axis=1), axis=0)
    df_rolling = df_props.rolling(window=5, center=True, min_periods=1).mean()

    df_rolling.plot(kind="area", stacked=True, colormap="viridis", figsize=(10,6))
    plt.title(f"{filename.capitalize()}: Anteile jeder Sentiment-Bewertung im Zeitverlauf (Gleitender 5-Jahres-Durchschnitt)")
    plt.xlabel("Jahr")
    plt.ylabel("Anteil")
    plt.legend(title="Sentiment")
    plt.grid(True)
    plt.savefig(f"{full_output_dir}/{filename}_{sentence_count}_stacked_sentiment_plot.png")
    plt.show()

def save_results_to_txt(filename, sentiment_counts):
    output_dir = get_code_path('output_results')
    os.makedirs(output_dir, exist_ok=True)
    total_sentiment = sentiment_counts.sum()

    with open(os.path.join(output_dir, f"{filename}_sentiment_counts.txt"), 'w') as f:
        f.write(f"{filename}\n")
        f.write(f"Total Sentences: {total_sentiment}\n")
        for sentiment in sorted(sentiment_counts.keys()):
            count = sentiment_counts[sentiment]
            f.write(f"{sentiment}: ({count}, {round(count / total_sentiment, 3)})\n")

def preprocess_data(filename, sample_size):
    print(f"Loading and preprocessing {filename}...")
    
    df = read_tsv(get_code_path(f"{filename}.tsv"))
    df = shorten_and_clean_df(df, filename, sample_size)
    df = remove_short_and_long_sentences(df)
    df = remove_duplicates(df)
    
    return df

def save_preprocessed_data(df, filename, sample_size):
    save_raw_text(df, filename, sample_size)
    save_filtered_text(df, filename, sample_size)
    print("Finished preprocessing")

def analyze_and_display_sentiment(df):
    df = analyze_sentiment(df)
    print("\nSample results:")
    print(df.head())
    
    sentiment_counts = df['Sentiment'].value_counts()
    print(f"\nSentiment distribution:")
    print(sentiment_counts)
    
    return df, sentiment_counts

def save_final_results(df, filename, sample_size, sentiment_counts):
    df_filtered = df[(df['Date'] >= GERMAN_EMPIRE_START) & (df['Date'] <= GERMAN_EMPIRE_END)]
    save_dataframe(df, filename, sample_size)
    save_results_to_txt(filename, sentiment_counts)
    plot_stacked_area_chart(df_filtered, filename=filename, sentence_count=sample_size)

def process_sentiment_analysis(filename=FILE_NAME, sample_size=DEFAULT_SAMPLE_SIZE):
    """Main function to process sentiment analysis on historical text data."""
    print(f"Processing {filename} with sample size: {sample_size}")
    
    df = preprocess_data(filename, sample_size)
    save_preprocessed_data(df, filename, sample_size)

    df, sentiment_counts = analyze_and_display_sentiment(df)
    save_final_results(df, filename, sample_size, sentiment_counts)

    print(f"\nAnalysis complete. Processed {len(df)} sentences.")


if __name__ == "__main__":
    process_sentiment_analysis()
