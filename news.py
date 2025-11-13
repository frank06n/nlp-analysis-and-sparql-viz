import streamlit as st
import requests
from bs4 import BeautifulSoup
import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from collections import Counter
import pandas as pd

# Download required NLTK data
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('tokenizers/punkt_tab')
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        nltk.download('punkt')
        nltk.download('punkt_tab')
        nltk.download('averaged_perceptron_tagger')

download_nltk_data()

# POS tag descriptions
POS_DESCRIPTIONS = {
    'NN': 'Noun, singular',
    'NNS': 'Noun, plural',
    'NNP': 'Proper noun, singular',
    'NNPS': 'Proper noun, plural',
    'VB': 'Verb, base form',
    'VBD': 'Verb, past tense',
    'VBG': 'Verb, gerund/present participle',
    'VBN': 'Verb, past participle',
    'VBP': 'Verb, non-3rd person singular present',
    'VBZ': 'Verb, 3rd person singular present',
    'JJ': 'Adjective',
    'JJR': 'Adjective, comparative',
    'JJS': 'Adjective, superlative',
    'RB': 'Adverb',
    'RBR': 'Adverb, comparative',
    'RBS': 'Adverb, superlative',
    'PRP': 'Personal pronoun',
    'PRP$': 'Possessive pronoun',
    'DT': 'Determiner',
    'IN': 'Preposition/subordinating conjunction',
    'CC': 'Coordinating conjunction',
    'CD': 'Cardinal number',
    'TO': 'to',
    'MD': 'Modal',
}

def fetch_article(url):
    """Fetch and extract text from news article URL"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(['script', 'style', 'nav', 'header', 'footer']):
            script.decompose()
        
        # Try to find article content (common patterns)
        article_text = ''
        
        # Try different selectors for article content
        selectors = [
            'article',
            '.article-content',
            '.story-content',
            '[class*="article"]',
            '[class*="content"]'
        ]
        
        for selector in selectors:
            content = soup.select_one(selector)
            if content:
                article_text = content.get_text(separator=' ', strip=True)
                break
        
        # If no article found, get all paragraph text
        if not article_text:
            paragraphs = soup.find_all('p')
            article_text = ' '.join([p.get_text(strip=True) for p in paragraphs])
        
        return article_text
    
    except Exception as e:
        return None, str(e)

def analyze_pos(text):
    """Analyze parts of speech in text"""
    # Tokenize and tag
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    
    # Count POS tags
    pos_counts = Counter([tag for word, tag in pos_tags])
    
    return pos_tags, pos_counts

# Streamlit UI
st.title("📰 News Article POS Analyzer")
st.markdown("Extract text from news articles and analyze parts of speech using NLTK")

# Input URL
url = st.text_input(
    "Enter News Article URL:",
    placeholder="https://example.com/article",
    value="https://timesofindia.indiatimes.com/business/india-business/train-americans-and-go-home-new-us-mantra-for-h-1b-visa-holders/articleshow/125303507.cms"
)

if st.button("Analyze Article", type="primary"):
    if url:
        with st.spinner("Fetching article..."):
            result = fetch_article(url)
            
            if isinstance(result, tuple):
                article_text, error = result
                st.error(f"Error fetching article: {error}")
            else:
                article_text = result
                
                if article_text:
                    st.success("Article fetched successfully!")
                    
                    # Show article preview
                    with st.expander("📄 Article Text Preview (first 500 characters)"):
                        st.write(article_text[:500] + "...")
                    
                    st.info(f"**Total characters:** {len(article_text)}")
                    
                    # Analyze POS
                    with st.spinner("Analyzing parts of speech..."):
                        pos_tags, pos_counts = analyze_pos(article_text)
                        
                        st.subheader("📊 Parts of Speech Analysis")
                        
                        # Create DataFrame for better display
                        df = pd.DataFrame([
                            {
                                'POS Tag': tag,
                                'Description': POS_DESCRIPTIONS.get(tag, 'Other'),
                                'Count': count
                            }
                            for tag, count in pos_counts.most_common()
                        ])
                        
                        # Display metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Words", len(pos_tags))
                        with col2:
                            st.metric("Unique POS Tags", len(pos_counts))
                        with col3:
                            st.metric("Most Common Tag", pos_counts.most_common(1)[0][0])
                        
                        # Display table
                        st.dataframe(df, use_container_width=True, hide_index=True)
                        
                        # Visualize top 10 POS tags
                        st.subheader("📈 Top 10 Most Frequent POS Tags")
                        top_10 = df.head(10)
                        st.bar_chart(top_10.set_index('POS Tag')['Count'])
                        
                        # Show sample words for each POS
                        with st.expander("🔍 Sample Words by POS Tag"):
                            pos_examples = {}
                            for word, tag in pos_tags:
                                if tag not in pos_examples:
                                    pos_examples[tag] = []
                                if len(pos_examples[tag]) < 5:
                                    pos_examples[tag].append(word)
                            
                            for tag in pos_counts.most_common(10):
                                tag_name = tag[0]
                                st.write(f"**{tag_name}** ({POS_DESCRIPTIONS.get(tag_name, 'Other')}): {', '.join(pos_examples.get(tag_name, []))}")
                else:
                    st.error("Could not extract text from the article. Please try a different URL.")
    else:
        st.warning("Please enter a valid URL")

# Info section
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This app uses:
    - **BeautifulSoup** to extract article text
    - **NLTK** for POS tagging
    - **Streamlit** for the interface
    
    ### How to use:
    1. Enter a news article URL
    2. Click "Analyze Article"
    3. View the POS analysis results
    
    ### Common POS Tags:
    - **NN/NNS**: Nouns
    - **VB/VBD/VBG**: Verbs
    - **JJ**: Adjectives
    - **RB**: Adverbs
    - **PRP**: Pronouns
    - **DT**: Determiners (a, the)
    - **IN**: Prepositions
    """)