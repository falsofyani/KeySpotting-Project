import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import torch
from transformers import ClapModel, ClapProcessor
import argparse

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = ClapModel.from_pretrained("laion/clap-htsat-fused")
processor = ClapProcessor.from_pretrained("laion/clap-htsat-fused")
model.to(device)
model.eval()


# ==================== EMBEDDING FUNCTIONS ====================

def extract_audio_embedding(wav_path, normalize=True):
    """Extract audio embedding from WAV file"""
    try:
        audio, sr = librosa.load(wav_path, sr=16000)
        inputs = processor(audio=[audio], sampling_rate=48000,
                           return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            embedding = model.get_audio_features(**inputs)

        embedding = embedding.cpu().numpy().squeeze()

        if normalize and np.linalg.norm(embedding) > 0:
            embedding = embedding / np.linalg.norm(embedding)

        return embedding
    except Exception as e:
        print(f"Error extracting audio embedding from {wav_path}: {e}")
        return None


def extract_text_embedding(text, normalize=True):
    """Extract text embedding from text"""
    try:
        text_input = processor(text=text, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            embedding = model.get_text_features(**text_input)

        embedding = embedding.cpu().numpy().squeeze()

        if normalize and np.linalg.norm(embedding) > 0:
            embedding = embedding / np.linalg.norm(embedding)

        return embedding
    except Exception as e:
        print(f"Error extracting text embedding for '{text}': {e}")
        return None


def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# ==================== DATASET LOADING ====================

def load_test_dataset(dataset_path="AR_recordings"):
    """Load the test dataset with correct/wrong structure"""
    audio_embeddings = {}

    for word in os.listdir(dataset_path):
        word_path = os.path.join(dataset_path, word)
        if not os.path.isdir(word_path):
            continue

        audio_embeddings[word] = {"correct": [], "wrong": []}

        # Load correct recordings
        correct_dir = os.path.join(word_path, "correct")
        if os.path.exists(correct_dir):
            for f in os.listdir(correct_dir):
                if f.endswith(('.wav', '.mp3', '.flac', '.m4a')):
                    path = os.path.join(correct_dir, f)
                    emb = extract_audio_embedding(path)
                    if emb is not None:
                        audio_embeddings[word]["correct"].append(emb)

        # Load wrong recordings
        wrong_dir = os.path.join(word_path, "wrong")
        if os.path.exists(wrong_dir):
            for f in os.listdir(wrong_dir):
                if f.endswith(('.wav', '.mp3', '.flac', '.m4a')):
                    path = os.path.join(wrong_dir, f)
                    emb = extract_audio_embedding(path)
                    if emb is not None:
                        audio_embeddings[word]["wrong"].append(emb)

        print(f"Loaded {word}: {len(audio_embeddings[word]['correct'])} correct, "
              f"{len(audio_embeddings[word]['wrong'])} wrong")

    return audio_embeddings


# ==================== KEYWORD DEFINITION METHODS ====================

def get_keyword_embeddings_from_text(keywords):
    """Get embeddings for keywords using text input"""
    print("\n" + "=" * 60)
    print("PROCESSING KEYWORDS AS TEXT")
    print("=" * 60)

    keyword_embeddings = {}

    for keyword in keywords:
        print(f"Processing keyword as text: '{keyword}'")
        text_emb = extract_text_embedding(keyword)
        if text_emb is not None:
            keyword_embeddings[keyword] = text_emb
            print(f"  ✓ Text embedding extracted")
        else:
            print(f"  ✗ Failed to extract text embedding")

    return keyword_embeddings


def get_keyword_embeddings_from_audio(audio_definitions_path="AR_keywords_definitions"):
    """Get embeddings for keywords using audio files"""
    print("\n" + "=" * 60)
    print("PROCESSING KEYWORDS FROM AUDIO FILES")
    print("=" * 60)

    keyword_embeddings = {}

    if not os.path.exists(audio_definitions_path):
        print(f"Error: Audio definitions path '{audio_definitions_path}' not found!")
        return keyword_embeddings

    # Option 1: Each keyword has its own folder with audio files
    for item in os.listdir(audio_definitions_path):
        item_path = os.path.join(audio_definitions_path, item)

        if os.path.isdir(item_path):
            # This is a keyword folder
            keyword = item
            print(f"\nProcessing keyword from audio folder: '{keyword}'")

            # Find all audio files in this folder
            audio_files = []
            for f in os.listdir(item_path):
                if f.endswith(('.wav', '.mp3', '.flac', '.m4a')):
                    audio_files.append(os.path.join(item_path, f))

            if not audio_files:
                print(f"  ✗ No audio files found for '{keyword}'")
                continue

            # Process each audio file
            embeddings = []
            for audio_file in audio_files:
                print(f"  Processing audio file: {os.path.basename(audio_file)}")
                emb = extract_audio_embedding(audio_file)
                if emb is not None:
                    embeddings.append(emb)

            if embeddings:
                # Use the FIRST audio file as the definition (not averaged)
                keyword_embeddings[keyword] = embeddings[0]
                print(f"  ✓ Audio embedding extracted ({len(embeddings)} files available)")

        elif item.endswith(('.wav', '.mp3', '.flac', '.m4a')):
            # This is a direct audio file (keyword name from filename)
            keyword = os.path.splitext(item)[0]
            print(f"\nProcessing keyword from audio file: '{keyword}'")

            emb = extract_audio_embedding(item_path)
            if emb is not None:
                keyword_embeddings[keyword] = emb
                print(f"  ✓ Audio embedding extracted")
            else:
                print(f"  ✗ Failed to extract audio embedding")

    return keyword_embeddings


# ==================== EVALUATION ====================

def evaluate_keywords(test_embeddings, keyword_embeddings, method_name="Text"):
    """
    Evaluate performance for all keywords
    Returns: dictionary with results
    """
    print(f"\n" + "=" * 60)
    print(f"EVALUATING WITH {method_name.upper()} DEFINITIONS")
    print("=" * 60)

    results = {}

    for keyword, definition_embedding in keyword_embeddings.items():
        if keyword not in test_embeddings:
            print(f"Warning: Keyword '{keyword}' not found in test dataset")
            continue

        if not test_embeddings[keyword]["correct"] or not test_embeddings[keyword]["wrong"]:
            print(f"Skipping '{keyword}': Need both correct and wrong samples")
            continue

        # Collect scores
        scores, labels = [], []

        # Positive examples (correct recordings)
        for emb in test_embeddings[keyword]["correct"]:
            scores.append(cosine_similarity(emb, definition_embedding))
            labels.append(1)

        # Negative examples (wrong recordings)
        for emb in test_embeddings[keyword]["wrong"]:
            scores.append(cosine_similarity(emb, definition_embedding))
            labels.append(0)

        if len(set(labels)) < 2:
            continue

        # Calculate ROC curve and metrics
        fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
        fnr = 1 - tpr  # False Negative Rate
        roc_auc = auc(fpr, tpr)

        # Find Equal Error Rate (EER)
        eer_idx = np.argmin(np.abs(fpr - fnr))
        eer = (fpr[eer_idx] + fnr[eer_idx]) / 2

        results[keyword] = {
            'auc': roc_auc,
            'eer': eer,
            'fpr': fpr,
            'fnr': fnr,
            'scores': scores,
            'labels': labels,
            'method': method_name
        }

        print(f"{keyword:15s} | AUC: {roc_auc:.3f} | EER: {eer:.3f}")

    return results


# ==================== PLOTTING ====================

def plot_single_method(results, method_name):
    """Plot DET curves for a single method"""
    if not results:
        print(f"No results to plot for {method_name}")
        return None

    plt.figure(figsize=(10, 8))

    # Sort by AUC for better visualization
    sorted_items = sorted(results.items(),
                          key=lambda x: x[1]['auc'],
                          reverse=True)

    # Create color gradient
    colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_items)))

    for (keyword, result), color in zip(sorted_items, colors):
        plt.plot(result['fpr'], result['fnr'],
                 color=color,
                 linewidth=2,
                 label=f"{keyword} (AUC={result['auc']:.2f}, EER={result['eer']:.2f})")

    # Add random baseline
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random')

    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("False Negative Rate", fontsize=12)
    plt.title(f"DET Curves – Keyword Spotting ({method_name} Definitions)",
              fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=9)
    plt.grid(True, alpha=0.2)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.tight_layout()

    # Save figure
    filename = f"det_curves_{method_name.lower()}_definitions.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {filename}")

    return plt.gcf()


def plot_comparison(text_results, audio_results):
    """Plot comparison between text and audio methods"""
    if not text_results or not audio_results:
        print("Cannot plot comparison: need both text and audio results")
        return None

    # Get common keywords
    common_keywords = set(text_results.keys()) & set(audio_results.keys())
    if not common_keywords:
        print("No common keywords between text and audio results")
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Text results
    sorted_text = sorted([(k, v) for k, v in text_results.items() if k in common_keywords],
                         key=lambda x: x[1]['auc'], reverse=True)

    colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_text)))

    for (keyword, result), color in zip(sorted_text, colors):
        ax1.plot(result['fpr'], result['fnr'],
                 color=color,
                 linewidth=2,
                 label=f"{keyword} (AUC={result['auc']:.2f})")

    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random')
    ax1.set_xlabel("False Positive Rate", fontsize=11)
    ax1.set_ylabel("False Negative Rate", fontsize=11)
    ax1.set_title("Text Definitions", fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=8)
    ax1.grid(True, alpha=0.2)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])

    # Plot 2: Audio results
    sorted_audio = sorted([(k, v) for k, v in audio_results.items() if k in common_keywords],
                          key=lambda x: x[1]['auc'], reverse=True)

    for (keyword, result), color in zip(sorted_audio, colors):
        ax2.plot(result['fpr'], result['fnr'],
                 color=color,
                 linewidth=2,
                 label=f"{keyword} (AUC={result['auc']:.2f})")

    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random')
    ax2.set_xlabel("False Positive Rate", fontsize=11)
    ax2.set_ylabel("False Negative Rate", fontsize=11)
    ax2.set_title("Audio Definitions", fontsize=12, fontweight='bold')
    ax2.legend(loc='best', fontsize=8)
    ax2.grid(True, alpha=0.2)
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])

    plt.suptitle("Comparison: Text vs Audio Keyword Definitions",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Save figure
    plt.savefig("comparison_text_vs_audio.png", dpi=150, bbox_inches='tight')
    print("Saved comparison plot to comparison_text_vs_audio.png")

    # Print comparison summary
    print("\n" + "=" * 60)
    print("PERFORMANCE COMPARISON SUMMARY")
    print("=" * 60)

    improvements = []
    for keyword in common_keywords:
        text_auc = text_results[keyword]['auc']
        audio_auc = audio_results[keyword]['auc']
        improvement = audio_auc - text_auc
        improvements.append(improvement)

        better = "Audio" if improvement > 0 else "Text" if improvement < 0 else "Equal"
        print(f"{keyword:15s} | Text AUC: {text_auc:.3f} | Audio AUC: {audio_auc:.3f} | "
              f"Diff: {improvement:+.3f} | Better: {better}")

    print(f"\nSummary:")
    print(f"Audio better for {sum(1 for x in improvements if x > 0)} keywords")
    print(f"Text better for {sum(1 for x in improvements if x < 0)} keywords")
    print(f"Equal for {sum(1 for x in improvements if x == 0)} keywords")
    print(f"Average AUC difference: {np.mean(improvements):+.3f} (positive favors audio)")

    return fig


# ==================== MAIN FUNCTION ====================

def main(use_text=True, use_audio=True, audio_definitions_path="AR_keywords_definitions"):
    """
    Main function to run keyword spotting evaluation

    Parameters:
    -----------
    use_text : bool
        Whether to evaluate using text definitions
    use_audio : bool
        Whether to evaluate using audio definitions
    audio_definitions_path : str
        Path to folder containing audio definitions
    """

    # Load test dataset
    print("Loading test dataset from AR_recordings...")
    test_embeddings = load_test_dataset("AR_recordings")

    # Get list of keywords from test dataset
    keywords = list(test_embeddings.keys())
    print(f"\nFound {len(keywords)} keywords in test dataset")

    text_results = None
    audio_results = None

    # Evaluate with text definitions
    if use_text:
        print("\n" + "=" * 60)
        print("RUNNING TEXT-BASED KEYWORD SPOTTING")
        print("=" * 60)

        # Get text embeddings for keywords
        text_keyword_embeddings = get_keyword_embeddings_from_text(keywords)

        if text_keyword_embeddings:
            # Evaluate
            text_results = evaluate_keywords(test_embeddings, text_keyword_embeddings, "Text")

            # Plot results
            if text_results:
                plot_single_method(text_results, "Text")
                plt.show()
        else:
            print("No text embeddings could be extracted!")

    # Evaluate with audio definitions
    if use_audio:
        print("\n" + "=" * 60)
        print("RUNNING AUDIO-BASED KEYWORD SPOTTING")
        print("=" * 60)

        # Get audio embeddings for keywords
        audio_keyword_embeddings = get_keyword_embeddings_from_audio(audio_definitions_path)

        if audio_keyword_embeddings:
            # Evaluate
            audio_results = evaluate_keywords(test_embeddings, audio_keyword_embeddings, "Audio")

            # Plot results
            if audio_results:
                plot_single_method(audio_results, "Audio")
                plt.show()
        else:
            print("No audio embeddings could be extracted!")

    # Plot comparison if both methods were used
    if use_text and use_audio and text_results and audio_results:
        plot_comparison(text_results, audio_results)
        plt.show()

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


# ==================== COMMAND LINE INTERFACE ====================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arabic Keyword Spotting System')
    parser.add_argument('--text-only', action='store_true',
                        help='Run only text-based keyword spotting')
    parser.add_argument('--audio-only', action='store_true',
                        help='Run only audio-based keyword spotting')
    parser.add_argument('--audio-path', type=str, default='AR_keywords_definitions',
                        help='Path to audio definitions folder')

    args = parser.parse_args()

    # Determine which methods to run
    if args.text_only and args.audio_only:
        print("Cannot use both --text-only and --audio-only. Running both methods.")
        use_text = True
        use_audio = True
    elif args.text_only:
        use_text = True
        use_audio = False
    elif args.audio_only:
        use_text = False
        use_audio = True
    else:
        # Default: run both
        use_text = True
        use_audio = True

    # Run main function
    main(use_text=use_text,
         use_audio=use_audio,
         audio_definitions_path=args.audio_path)