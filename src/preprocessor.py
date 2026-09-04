import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Safe downloader function for NLTK resources
def download_nltk_resources():
    resources = {
        'stopwords': 'corpora/stopwords',
        'wordnet': 'corpora/wordnet',
        'punkt': 'tokenizers/punkt',
        'omw-1.4': 'corpora/omw-1.4'
    }
    for res_name, res_path in resources.items():
        try:
            nltk.data.find(res_path)
        except LookupError:
            print(f"Downloading missing NLTK dependency: '{res_name}'...")
            nltk.download(res_name, quiet=True)

# Run downloader on module import
download_nltk_resources()

class Preprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Keep sentiment-carrying negation words in stopwords if we want to retain context
        # (e.g., 'not', 'no', 'but', 'nor' can change sentiment drastically)
        negation_words = {'not', 'no', 'but', 'nor', 'against', 'only', 'never', 'off'}
        self.stop_words = self.stop_words - negation_words

    def clean_text(self, text):
        """Strips structural noise such as URLs, HTML tags, @mentions, and excess whitespaces."""
        if not isinstance(text, str):
            return ""
            
        # 1. Decode HTML entities (e.g. &amp; -> &, &lt; -> <)
        text = re.sub(r'&[a-zA-Z0-9#]+;', ' ', text)
        
        # 2. Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # 3. Remove @UserMentions
        text = re.sub(r'@\w+', '', text)
        
        # 4. Clean extra spaces, newlines, and tabs
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def preprocess(self, text, remove_stopwords=True, lemmatize=True):
        """Main pipeline: cleans, lowercases, tokenizes, removes stopwords, and lemmatizes."""
        cleaned = self.clean_text(text)
        if not cleaned:
            return []
            
        # Lowercase
        lowercased = cleaned.lower()
        
        # Tokenize (using standard word tokenizer)
        try:
            tokens = word_tokenize(lowercased)
        except LookupError:
            # Fallback split if punkt has any issue
            tokens = lowercased.split()
            
        processed_tokens = []
        for token in tokens:
            # Filter non-alphanumeric tokens (keep words, numbers, and basic emojis)
            # We filter punctuation strings but keep words containing alphanumeric chars
            if not re.match(r'^[^\w\s]+$', token):
                # Clean token from outer punctuation
                token = re.sub(r'^[^\w]+|[^\w]+$', '', token)
                if not token:
                    continue
                    
                # Stopwords filter
                if remove_stopwords and token in self.stop_words:
                    continue
                    
                # Lemmatizer
                if lemmatize:
                    token = self.lemmatizer.lemmatize(token)
                    
                processed_tokens.append(token)
                
        return processed_tokens

    def preprocess_as_string(self, text, remove_stopwords=True, lemmatize=True):
        """Convenience method returning a single space-separated string instead of a token list."""
        tokens = self.preprocess(text, remove_stopwords, lemmatize)
        return " ".join(tokens)
