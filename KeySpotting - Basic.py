# install necessary packages
# pip install numpy librosa matplotlib scipy scikit-learn

# The Arabic dataset used from https://www.kaggle.com/datasets/abdulkaderghandoura/arabic-speech-commands-dataset

# import necessary libraries
import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from sklearn.metrics import det_curve

# Define Arabic Words for Key Spotting
ARABIC_KEYWORDS = {
    "نعم": "yes",
    "لا": "no",
    "توقف": "stop",
    "ابدأ": "start",
    "افتح": "open",
    "اغلق": "close"
}

# Define phoneme representations for Arabic keywords
arabic_phonemes = {
    "نعم": ["n", "a", "a", "a", "m"],
    "لا": ["l", "a"],
    "توقف": ["t", "w", "q", "q", "a", "f"],
    "ابدأ": ["e", "b", "d", "a"],
    "افتح": ["e", "f", "t", "a", "ħ"],
    "اغلق": ["a", "g", "l", "e", "q"]
}

# Function to extract MFCC features from audio file
def extract_mfcc(wav_path, sr=16000, n_mfcc=40):
    """
    Extract MFCC features from an audio file.
    yields a 2-D array of shape (time, n_mfcc)
    """
    y, sr = librosa.load(wav_path, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfcc.T  # shape: (time, features)

# Simple mean pooling → lightweight baseline embedding.
def audio_embedding(mfcc):
    """Return a 1-D embedding vector for an audio file's MFCCs.

    Accepts either a 2-D array of frames x coeffs (typical MFCC output) or
    a pre-pooled 1-D vector. This avoids accidentally returning a scalar when
    a 1-D array is passed.
    """
    mean = np.mean(mfcc, axis=0)
    std = np.std(mfcc, axis=0)
    emb = np.concatenate([mean, std])
    return emb.reshape(-1)

# Load Dataset
def load_dataset(base_dir):
    """Load audio files and their labels from a directory structure."""
    X = []
    y = []

    for label in os.listdir(base_dir):
        label_dir = os.path.join(base_dir, label)
        if not os.path.isdir(label_dir):
            continue

        for wav in os.listdir(label_dir):
            if wav.endswith(".wav"):
                path = os.path.join(label_dir, wav)
                mfcc = extract_mfcc(path)
                emb = audio_embedding(mfcc)
                X.append(emb)
                y.append(label)

    return np.array(X), np.array(y)

# Build Reference Keyword Embeddings
def build_audio_keyword_embeddings(X, y):
    """
    Build average embeddings for each keyword.
    Returns a dict mapping keyword labels to their average embeddings.
    """
    keyword_embs = {}
    for label in np.unique(y):
        keyword_embs[label] = np.mean(X[y == label], axis=0)
    return keyword_embs

np.random.seed(42)

phoneme_set = sorted({p for phones in arabic_phonemes.values() for p in phones})
phoneme_embeddings = {p: np.random.randn(80) for p in phoneme_set}

def text_embedding(word):
    """
    Return a text-based embedding for an Arabic word based on its phonemes.
    Each phoneme is represented by a random vector, and the word embedding is the mean of these.
    """
    phones = arabic_phonemes[word]
    embs = [phoneme_embeddings[p] for p in phones]
    return np.mean(embs, axis=0).reshape(-1)

def build_text_keyword_embeddings(keywords):
    """
    Build text-based embeddings for each keyword.
    Returns a dict mapping keyword labels to their text-based embeddings.
    """
    return {word: text_embedding(word) for word in keywords}


# Compute Detection Scores 
# Lower distance = more confident detection.
def compute_scores(X, y_true, keyword_embs):
    """
    Compute cosine distance scores between each embedding and keyword embeddings.
    Returns scores and corresponding binary labels (1 for match, 0 for non-match).
    """
    scores = []
    labels = []

    for emb, true_label in zip(X, y_true):
        for keyword, ref_emb in keyword_embs.items():
            dist = cosine(emb, ref_emb)
            scores.append(dist)
            labels.append(1 if keyword == true_label else 0)

    return np.array(scores), np.array(labels)

# Plot DET Curve
def plot_det(scores, labels, title="DET Curve (Arabic KWS)"):
    """
    Plot Detection Error Tradeoff (DET) curve.
    """
    fpr, fnr, _ = det_curve(labels, scores)

    plt.figure()
    plt.plot(fpr, fnr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("Missed Detection Rate")
    plt.title(title)
    plt.grid(True)
    plt.show()

# Plot DET Curve per word
def plot_det_per_keyword(X, y_true, keyword_embs):
    """
    Plot DET curves for each keyword.
    """
    plt.figure(figsize=(8, 6))

    for keyword in keyword_embs.keys():
        scores = []
        labels = []

        ref_emb = keyword_embs[keyword].reshape(-1)

        for emb, true_label in zip(X, y_true):
            emb = emb.reshape(-1)
            dist = cosine(emb, ref_emb)

            scores.append(dist)
            labels.append(1 if true_label == keyword else 0)

        scores = np.array(scores)
        labels = np.array(labels)

        # Skip if not enough positives or negatives
        if len(np.unique(labels)) < 2:
            continue

        fpr, fnr, _ = det_curve(labels, scores)
        plt.plot(fpr, fnr, label=keyword)

    plt.xlabel("False Positive Rate")
    plt.ylabel("Missed Detection Rate")
    plt.title("DET Curves per Arabic Keyword")
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    # Path to dataset
    DATA_DIR = "Arabic_Words"

    # Load data
    X, y = load_dataset(DATA_DIR)

    # Build keyword embeddings
    keyword_embs = build_audio_keyword_embeddings(X, y)

    # Compute detection scores
    scores, labels = compute_scores(X, y, keyword_embs)

    # Plot DET curve
    plot_det(scores, labels)

    # Plot DET curves per keyword
    plot_det_per_keyword(X, y, keyword_embs)

    # Using text-based embeddings
    text_keyword_embs = build_text_keyword_embeddings(ARABIC_KEYWORDS)
    scores_text, labels_text = compute_scores(X, y, text_keyword_embs)

    # Plot DET curve for text embeddings
    plot_det(scores_text, labels_text, "DET Curve – Arabic KWS (Text Embeddings)")

if __name__ == "__main__":
    main()