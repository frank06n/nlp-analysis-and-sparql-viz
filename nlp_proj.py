import nltk
import spacy
from nltk.corpus import brown
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import numpy as np

# Download required NLTK data
print("Downloading required NLTK data...")
nltk.download('brown')
nltk.download('universal_tagset')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
print("NLTK data downloaded successfully!")

# Load spaCy model
print("Loading spaCy model...")
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    print("Downloading spaCy model...")
    import os
    os.system('python -m spacy download en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')

# Universal POS tag mapping for spaCy
SPACY_TO_UNIVERSAL = {
    'ADJ': 'ADJ', 'ADP': 'ADP', 'ADV': 'ADV', 'AUX': 'VERB',
    'CONJ': 'CONJ', 'CCONJ': 'CONJ', 'DET': 'DET', 'INTJ': 'X',
    'NOUN': 'NOUN', 'NUM': 'NUM', 'PART': 'PRT', 'PRON': 'PRON',
    'PROPN': 'NOUN', 'PUNCT': '.', 'SCONJ': 'CONJ', 'SYM': 'X',
    'VERB': 'VERB', 'X': 'X', 'SPACE': 'X'
}

def get_brown_sentences(n_sentences=1000):
    """Extract sentences with universal POS tags from Brown corpus"""
    print(f"\nExtracting {n_sentences} sentences from Brown corpus...")
    tagged_sents = brown.tagged_sents(tagset='universal')[:n_sentences]
    return tagged_sents

def tag_with_nltk(sentences):
    """Tag sentences using NLTK and convert to universal tagset"""
    print("\nTagging with NLTK...")
    nltk_tagged = []
    for sent in sentences:
        words = [word for word, tag in sent]
        tags = nltk.pos_tag(words, tagset='universal')
        nltk_tagged.append(tags)
    return nltk_tagged

def tag_with_spacy(sentences):
    """Tag sentences using spaCy and convert to universal tagset"""
    print("Tagging with spaCy...")
    spacy_tagged = []
    for sent in sentences:
        words = [word for word, tag in sent]
        text = ' '.join(words)
        doc = nlp(text)
        tags = [(token.text, SPACY_TO_UNIVERSAL.get(token.pos_, 'X')) 
                for token in doc]
        spacy_tagged.append(tags)
    return spacy_tagged

def calculate_accuracy(true_tags, pred_tags):
    """Calculate accuracy metrics"""
    true_flat = [tag for sent in true_tags for word, tag in sent]
    pred_flat = [tag for sent in pred_tags for word, tag in sent]
    
    # Ensure same length (handle tokenization differences)
    min_len = min(len(true_flat), len(pred_flat))
    true_flat = true_flat[:min_len]
    pred_flat = pred_flat[:min_len]
    
    accuracy = accuracy_score(true_flat, pred_flat)
    return accuracy, true_flat, pred_flat

def compare_taggers(n_sentences=1000):
    """Main comparison function"""
    print("="*60)
    print("POS TAGGER COMPARISON: NLTK vs spaCy")
    print("="*60)
    
    # Get Brown corpus data
    brown_sents = get_brown_sentences(n_sentences)
    
    # Tag with both systems
    nltk_tags = tag_with_nltk(brown_sents)
    spacy_tags = tag_with_spacy(brown_sents)
    
    # Calculate accuracies
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    nltk_acc, true_flat, nltk_flat = calculate_accuracy(brown_sents, nltk_tags)
    spacy_acc, _, spacy_flat = calculate_accuracy(brown_sents, spacy_tags)
    
    print(f"\nNLTK Accuracy:  {nltk_acc:.4f} ({nltk_acc*100:.2f}%)")
    print(f"spaCy Accuracy: {spacy_acc:.4f} ({spacy_acc*100:.2f}%)")
    print(f"\nDifference: {abs(nltk_acc - spacy_acc):.4f}")
    
    # Detailed classification reports
    print("\n" + "="*60)
    print("NLTK CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(true_flat, nltk_flat, zero_division=0))
    
    print("\n" + "="*60)
    print("spaCy CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(true_flat, spacy_flat, zero_division=0))
    
    # Per-tag accuracy comparison
    print("\n" + "="*60)
    print("PER-TAG ACCURACY COMPARISON")
    print("="*60)
    
    unique_tags = sorted(set(true_flat))
    results = []
    
    for tag in unique_tags:
        true_tag_indices = [i for i, t in enumerate(true_flat) if t == tag]
        if len(true_tag_indices) > 0:
            nltk_correct = sum([1 for i in true_tag_indices if nltk_flat[i] == tag])
            spacy_correct = sum([1 for i in true_tag_indices if spacy_flat[i] == tag])
            total = len(true_tag_indices)
            
            results.append({
                'Tag': tag,
                'Count': total,
                'NLTK Acc': f"{(nltk_correct/total)*100:.2f}%",
                'spaCy Acc': f"{(spacy_correct/total)*100:.2f}%",
                'Better': 'NLTK' if nltk_correct > spacy_correct else 'spaCy' if spacy_correct > nltk_correct else 'Tie'
            })
    
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    
    # Example comparisons
    print("\n" + "="*60)
    print("EXAMPLE DISAGREEMENTS (First 5)")
    print("="*60)
    
    disagreements = []
    for i, (true_tag, nltk_tag, spacy_tag) in enumerate(zip(true_flat, nltk_flat, spacy_flat)):
        if nltk_tag != spacy_tag:
            disagreements.append({
                'Index': i,
                'True': true_tag,
                'NLTK': nltk_tag,
                'spaCy': spacy_tag
            })
    
    if disagreements:
        for d in disagreements[:5]:
            print(f"\nPosition {d['Index']}:")
            print(f"  True Tag:  {d['True']}")
            print(f"  NLTK:      {d['NLTK']} {'✓' if d['NLTK'] == d['True'] else '✗'}")
            print(f"  spaCy:     {d['spaCy']} {'✓' if d['spaCy'] == d['True'] else '✗'}")
        
        print(f"\nTotal disagreements: {len(disagreements)} out of {len(true_flat)} tokens")
    
    return {
        'nltk_accuracy': nltk_acc,
        'spacy_accuracy': spacy_acc,
        'per_tag_results': df
    }

if __name__ == "__main__":
    # Run comparison on 1000 sentences
    results = compare_taggers(n_sentences=1000)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Winner: {'NLTK' if results['nltk_accuracy'] > results['spacy_accuracy'] else 'spaCy'}")
    print(f"by {abs(results['nltk_accuracy'] - results['spacy_accuracy'])*100:.2f} percentage points")