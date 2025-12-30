# install necessary packages
# pip install numpy librosa matplotlib scipy scikit-learn torch torchaudio transformers phonemizer

# The Arabic dataset used from https://www.kaggle.com/datasets/abdulkaderghandoura/arabic-speech-commands-dataset

# import necessary libraries
import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from sklearn.metrics import det_curve
import torch
import torchaudio
from transformers import Wav2Vec2Model, Wav2Vec2Processor
from phonemizer import phonemize

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Replace custom GRU encoders with pretrained wav2vec 2.0 model 
# Load pre-trained Wav2Vec2 model and processor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
audio_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(device)
audio_model.eval()

# Function to extract audio embedding using Wav2Vec2
def extract_audio_embedding(wav_path):
    audio, sr = librosa.load(wav_path, sr=16000)
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = audio_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy().squeeze()

# Function to convert text to phonemes (with fallback if espeak not installed)
def text_to_phonemes(word):
    try:
        phones = phonemize(
            word,
            language="ar",
            backend="espeak",
            strip=True,
            with_stress=False
        )
        # phonemize returns a string; split into phoneme tokens
        if isinstance(phones, str):
            phones = phones.split()
        # if phonemizer returned nothing, fallback to character tokens
        if not phones:
            return list(word)
        return phones
    except Exception as e:
        # espeak (or another phonemizer backend) may not be available on the system
        print(f"Warning: phonemizer backend not available ({e}); falling back to characters.")
        return list(word)

# simple phoneme embedding table (learned or random)
phoneme_vocab = {}
def phoneme_embedding(p):
    if p not in phoneme_vocab:
        phoneme_vocab[p] = np.random.randn(768)
    return phoneme_vocab[p]

# Function to extract text embedding from phonemes
def extract_text_embedding(word):
    phones = text_to_phonemes(word)
    emb = np.mean([phoneme_embedding(p) for p in phones], axis=0)
    return emb

# Cosine distance function
def cosine_distance(a, b):
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Plot DET curve for a given keyword
def plot_det(keyword, audio_embeddings, text_embedding):
    scores, labels = [], []

    for k, emb_list in audio_embeddings.items():
        for emb in emb_list:
            scores.append(cosine_distance(emb, text_embedding))
            labels.append(1 if k == keyword else 0)

    fpr, fnr, _ = det_curve(labels, scores, pos_label=1)
    plt.plot(fpr, fnr, label=keyword)

def main():
    audio_embeddings = {}

    # Path to dataset
    DATA_DIR = "Arabic_Words"

    # Load audio embeddings
    for word in os.listdir(DATA_DIR):
        audio_embeddings[word] = []
        for f in os.listdir(os.path.join(DATA_DIR, word)):
            path = os.path.join(DATA_DIR, word, f)
            audio_embeddings[word].append(extract_audio_embedding(path))

    # Plot DET curves for each keyword
    plt.figure()
    for word in audio_embeddings:
        text_emb = extract_text_embedding(word)
        plot_det(word, audio_embeddings, text_emb)

    plt.xlabel("False Positive Rate")
    plt.ylabel("Missed Detection Rate")
    plt.title("DET Curves – Arabic Open-Vocabulary KWS")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()