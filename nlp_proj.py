"""
Complete POS Tagger Comparison: NLTK vs spaCy
Single file Streamlit application with built-in setup
Run with: streamlit run this_script.py
"""

import streamlit as st
import sys
import os

# Setup and imports section
def setup_environment():
    """Setup all required packages and data"""
    import subprocess
    
    status = st.empty()
    progress = st.progress(0)
    
    try:
        status.text("Checking dependencies...")
        progress.progress(10)
        
        # Import required packages
        try:
            import nltk
            import spacy
            from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
            import pandas as pd
            import numpy as np
            import plotly.graph_objects as go
            import plotly.express as px
        except ImportError as e:
            status.error(f"Missing package: {e}")
            st.info("Please install: pip install streamlit nltk spacy scikit-learn pandas plotly")
            return False
        
        progress.progress(25)
        status.text("Setting up SSL bypass...")
        
        # SSL bypass for NLTK downloads
        import ssl
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context
        
        progress.progress(40)
        status.text("Downloading NLTK data...")
        
        # Download NLTK data
        packages = ['brown', 'universal_tagset', 'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng']
        for i, package in enumerate(packages):
            try:
                nltk.download(package, quiet=True)
            except:
                st.warning(f"Failed to download {package} - trying alternative method...")
                try:
                    nltk.download(package, quiet=False, raise_on_error=True)
                except:
                    pass
            progress.progress(40 + (i+1) * 10)
        
        progress.progress(70)
        status.text("Loading spaCy model...")
        
        # Load or download spaCy model
        try:
            nlp = spacy.load('en_core_web_sm')
        except OSError:
            status.text("Downloading spaCy model (this may take a few minutes)...")
            subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], 
                         capture_output=True)
            nlp = spacy.load('en_core_web_sm')
        
        progress.progress(90)
        status.text("Verifying Brown corpus...")
        
        # Test Brown corpus access
        from nltk.corpus import brown
        test = list(brown.tagged_sents(tagset='universal')[:5])
        
        progress.progress(100)
        status.success("✅ Setup complete!")
        progress.empty()
        status.empty()
        
        return True
        
    except Exception as e:
        status.error(f"Setup failed: {str(e)}")
        st.error("""
        ### Manual Setup Required
        
        Please run these commands:
        ```bash
        pip install nltk spacy scikit-learn pandas plotly
        python -m spacy download en_core_web_sm
        ```
        
        Then in Python:
        ```python
        import nltk
        nltk.download('brown')
        nltk.download('universal_tagset')
        nltk.download('averaged_perceptron_tagger')
        ```
        """)
        return False

# Only run setup once
if 'setup_complete' not in st.session_state:
    st.title("🏷️ POS Tagger Comparison Setup")
    st.info("Setting up required packages and data... This is a one-time process.")
    
    if setup_environment():
        st.session_state.setup_complete = True
        st.rerun()
    else:
        st.stop()

# Main application imports (after setup)
import nltk
import spacy
from nltk.corpus import brown
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="POS Tagger Comparison",
    page_icon="🏷️",
    layout="wide"
)

# Universal POS tag mapping for spaCy
SPACY_TO_UNIVERSAL = {
    'ADJ': 'ADJ', 'ADP': 'ADP', 'ADV': 'ADV', 'AUX': 'VERB',
    'CONJ': 'CONJ', 'CCONJ': 'CONJ', 'DET': 'DET', 'INTJ': 'X',
    'NOUN': 'NOUN', 'NUM': 'NUM', 'PART': 'PRT', 'PRON': 'PRON',
    'PROPN': 'NOUN', 'PUNCT': '.', 'SCONJ': 'CONJ', 'SYM': 'X',
    'VERB': 'VERB', 'X': 'X', 'SPACE': 'X'
}

@st.cache_resource
def load_spacy_model():
    """Load spaCy model"""
    return spacy.load('en_core_web_sm')

def get_brown_sentences(n_sentences=1000):
    """Extract sentences with universal POS tags from Brown corpus"""
    tagged_sents = brown.tagged_sents(tagset='universal')[:n_sentences]
    # Convert to plain lists to avoid pickling issues
    return [list(sent) for sent in tagged_sents]

def tag_with_nltk(sentences):
    """Tag sentences using NLTK and convert to universal tagset"""
    nltk_tagged = []
    for sent in sentences:
        words = [word for word, tag in sent]
        tags = nltk.pos_tag(words, tagset='universal')
        nltk_tagged.append(tags)
    return nltk_tagged

def tag_with_spacy(sentences, nlp):
    """Tag sentences using spaCy and convert to universal tagset"""
    spacy_tagged = []
    for sent in sentences:
        words = [word for word, tag in sent]
        text = ' '.join(words)
        doc = nlp(text)
        tags = [(token.text, SPACY_TO_UNIVERSAL.get(token.pos_, 'X')) 
                for token in doc]
        spacy_tagged.append(tags)
    return spacy_tagged

def calculate_metrics(true_tags, pred_tags):
    """Calculate accuracy metrics"""
    true_flat = [tag for sent in true_tags for word, tag in sent]
    pred_flat = [tag for sent in pred_tags for word, tag in sent]
    
    min_len = min(len(true_flat), len(pred_flat))
    true_flat = true_flat[:min_len]
    pred_flat = pred_flat[:min_len]
    
    accuracy = accuracy_score(true_flat, pred_flat)
    return accuracy, true_flat, pred_flat

def create_confusion_matrix_plot(true_flat, pred_flat, tagger_name):
    """Create confusion matrix heatmap"""
    tags = sorted(set(true_flat))
    cm = confusion_matrix(true_flat, pred_flat, labels=tags)
    
    fig = px.imshow(
        cm,
        labels=dict(x="Predicted", y="True", color="Count"),
        x=tags,
        y=tags,
        color_continuous_scale="Blues",
        title=f"{tagger_name} Confusion Matrix"
    )
    fig.update_layout(height=500)
    return fig

def create_accuracy_comparison_chart(df):
    """Create bar chart comparing per-tag accuracy"""
    fig = go.Figure()
    
    nltk_acc = [float(x.strip('%')) for x in df['NLTK Acc']]
    spacy_acc = [float(x.strip('%')) for x in df['spaCy Acc']]
    
    fig.add_trace(go.Bar(
        name='NLTK',
        x=df['Tag'],
        y=nltk_acc,
        marker_color='lightblue'
    ))
    
    fig.add_trace(go.Bar(
        name='spaCy',
        x=df['Tag'],
        y=spacy_acc,
        marker_color='lightcoral'
    ))
    
    fig.update_layout(
        title="Per-Tag Accuracy Comparison",
        xaxis_title="POS Tag",
        yaxis_title="Accuracy (%)",
        barmode='group',
        height=400
    )
    
    return fig

def run_analysis(n_sentences, nlp):
    """Main analysis function"""
    # Get data
    brown_sents = get_brown_sentences(n_sentences)
    
    # Create progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("Tagging with NLTK...")
        progress_bar.progress(25)
        nltk_tags = tag_with_nltk(brown_sents)
        
        status_text.text("Tagging with spaCy...")
        progress_bar.progress(50)
        spacy_tags = tag_with_spacy(brown_sents, nlp)
        
        status_text.text("Calculating metrics...")
        progress_bar.progress(75)
        
        # Calculate metrics
        nltk_acc, true_flat, nltk_flat = calculate_metrics(brown_sents, nltk_tags)
        spacy_acc, _, spacy_flat = calculate_metrics(brown_sents, spacy_tags)
        
        # Per-tag results
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
        
        # Find disagreements
        disagreements = []
        for i, (true_tag, nltk_tag, spacy_tag) in enumerate(zip(true_flat, nltk_flat, spacy_flat)):
            if nltk_tag != spacy_tag:
                disagreements.append({
                    'Position': i,
                    'True Tag': true_tag,
                    'NLTK': nltk_tag,
                    'NLTK Correct': '✓' if nltk_tag == true_tag else '✗',
                    'spaCy': spacy_tag,
                    'spaCy Correct': '✓' if spacy_tag == true_tag else '✗'
                })
        
        progress_bar.progress(100)
        status_text.text("Analysis complete!")
        progress_bar.empty()
        status_text.empty()
        
        return {
            'nltk_acc': nltk_acc,
            'spacy_acc': spacy_acc,
            'df': df,
            'disagreements': disagreements,
            'true_flat': true_flat,
            'nltk_flat': nltk_flat,
            'spacy_flat': spacy_flat,
            'n_sentences': n_sentences
        }
    
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"Analysis failed: {str(e)}")
        return None

# Main app
st.title("🏷️ POS Tagger Comparison: NLTK vs spaCy")
st.markdown("Compare the performance of NLTK and spaCy part-of-speech taggers on the Brown corpus")

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Settings")
    n_sentences = st.slider(
        "Number of sentences to analyze",
        min_value=100,
        max_value=2000,
        value=1000,
        step=100
    )
    
    run_analysis_btn = st.button("🚀 Run Analysis", type="primary", use_container_width=True)
    
    if st.button("🔄 Reset", use_container_width=True):
        st.session_state.results = None
        st.rerun()
    
    st.divider()
    st.markdown("""
    ### About
    This tool compares NLTK and spaCy POS taggers using:
    - **Dataset**: Brown Corpus
    - **Tagset**: Universal POS Tags
    - **Metrics**: Accuracy, Precision, Recall, F1-Score
    
    ### POS Tags
    - **NOUN**: Nouns
    - **VERB**: Verbs
    - **ADJ**: Adjectives
    - **ADV**: Adverbs
    - **PRON**: Pronouns
    - **DET**: Determiners
    - **ADP**: Adpositions
    - **NUM**: Numbers
    - **CONJ**: Conjunctions
    - **PRT**: Particles
    - **.**: Punctuation
    - **X**: Other
    """)

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = None

# Load spaCy model
nlp = load_spacy_model()

# Run analysis
if run_analysis_btn or st.session_state.results is not None:
    if run_analysis_btn:
        with st.spinner(f"Analyzing {n_sentences} sentences..."):
            st.session_state.results = run_analysis(n_sentences, nlp)
    
    if st.session_state.results is None:
        st.error("Analysis failed. Please try again.")
        st.stop()
    
    # Display results
    results = st.session_state.results
    
    # Overall accuracy
    st.header("📊 Overall Results")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "NLTK Accuracy",
            f"{results['nltk_acc']*100:.2f}%"
        )
    
    with col2:
        st.metric(
            "spaCy Accuracy",
            f"{results['spacy_acc']*100:.2f}%"
        )
    
    with col3:
        diff = abs(results['nltk_acc'] - results['spacy_acc']) * 100
        winner = "NLTK" if results['nltk_acc'] > results['spacy_acc'] else "spaCy"
        st.metric(
            "Winner",
            winner,
            delta=f"{diff:.2f}pp"
        )
    
    # Per-tag comparison
    st.header("📈 Per-Tag Accuracy Comparison")
    fig = create_accuracy_comparison_chart(results['df'])
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed table
    st.subheader("Detailed Per-Tag Results")
    st.dataframe(
        results['df'],
        use_container_width=True,
        hide_index=True
    )
    
    # Confusion matrices
    st.header("🔥 Confusion Matrices")
    col1, col2 = st.columns(2)
    
    with col1:
        fig_nltk = create_confusion_matrix_plot(
            results['true_flat'],
            results['nltk_flat'],
            "NLTK"
        )
        st.plotly_chart(fig_nltk, use_container_width=True)
    
    with col2:
        fig_spacy = create_confusion_matrix_plot(
            results['true_flat'],
            results['spacy_flat'],
            "spaCy"
        )
        st.plotly_chart(fig_spacy, use_container_width=True)
    
    # Disagreements
    st.header("⚠️ Disagreements Between Taggers")
    st.markdown(f"**Total disagreements:** {len(results['disagreements'])} out of {len(results['true_flat'])} tokens ({len(results['disagreements'])/len(results['true_flat'])*100:.2f}%)")
    
    if results['disagreements']:
        st.subheader("Sample Disagreements (First 20)")
        disagreement_df = pd.DataFrame(results['disagreements'][:20])
        st.dataframe(
            disagreement_df,
            use_container_width=True,
            hide_index=True
        )
    
    # Classification reports
    with st.expander("📋 Detailed Classification Reports"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("NLTK Classification Report")
            report = classification_report(
                results['true_flat'],
                results['nltk_flat'],
                zero_division=0,
                output_dict=True
            )
            st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)
        
        with col2:
            st.subheader("spaCy Classification Report")
            report = classification_report(
                results['true_flat'],
                results['spacy_flat'],
                zero_division=0,
                output_dict=True
            )
            st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)
    
    # Download results
    st.divider()
    
    # Create downloadable CSV
    csv = results['df'].to_csv(index=False)
    st.download_button(
        label="📥 Download Results as CSV",
        data=csv,
        file_name="pos_tagger_comparison_results.csv",
        mime="text/csv"
    )

else:
    st.info("👈 Configure settings in the sidebar and click 'Run Analysis' to begin")
    
    # Show example
    st.markdown("""
    ### What is POS Tagging?
    
    Part-of-speech (POS) tagging is the process of marking up words in a text with their corresponding part of speech (noun, verb, adjective, etc.).
    
    **Example:**
    - **Sentence**: "The quick brown fox jumps"
    - **Tagged**: The/DET quick/ADJ brown/ADJ fox/NOUN jumps/VERB
    
    ### How This Tool Works
    
    1. **Select** the number of sentences from the Brown corpus
    2. **Click** "Run Analysis" to compare NLTK and spaCy taggers
    3. **View** accuracy metrics, confusion matrices, and disagreements
    4. **Download** results for further analysis
    
    ### Why Compare Taggers?
    
    Different POS taggers use different algorithms and training data:
    - **NLTK**: Uses the averaged perceptron tagger
    - **spaCy**: Uses deep learning models
    
    This comparison helps you understand which tagger works best for your use case!
    """)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    <p>Built with Streamlit • NLTK • spaCy • scikit-learn</p>
    <p>Using Brown Corpus with Universal POS Tagset</p>
</div>
""", unsafe_allow_html=True)