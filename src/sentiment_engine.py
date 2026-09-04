import os
import pickle
import yaml
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from preprocessor import Preprocessor

class SentimentEngine:
    def __init__(self, config_path="config/config.yaml"):
        # Load configuration
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
            
        self.model_type = self.config.get("active_model", "vader").lower()
        self.preprocessor = Preprocessor()
        
        # Placeholders for loaded models
        self.vader_analyzer = None
        self.vectorizer = None
        self.classifier = None
        self.hf_pipeline = None
        
        # Load the selected model on initialization
        self._load_model()

    def _load_model(self):
        """Loads model files based on selection in config.yaml."""
        if self.model_type == "vader":
            try:
                nltk.data.find("sentiment/vader_lexicon")
            except LookupError:
                print("Downloading VADER lexicon NLTK dependency...")
                nltk.download("vader_lexicon", quiet=True)
            self.vader_analyzer = SentimentIntensityAnalyzer()
            print("Loaded Lexicon Model: NLTK VADER Analyzer.")
            
        elif self.model_type == "tfidf_logistic":
            vec_path = self.config["model_paths"]["vectorizer"]
            clf_path = self.config["model_paths"]["classifier"]
            
            if not os.path.exists(vec_path) or not os.path.exists(clf_path):
                print(f"WARNING: Model weights not found at '{vec_path}' or '{clf_path}'.")
                print("Please run 'src/model_trainer.py' first to train the ML models.")
                # Fallback to VADER if ML weights are missing
                self.model_type = "vader"
                self._load_model()
            else:
                with open(vec_path, 'rb') as f:
                    self.vectorizer = pickle.load(f)
                with open(clf_path, 'rb') as f:
                    self.classifier = pickle.load(f)
                print("Loaded Classical ML Model: TF-IDF + Logistic Regression.")
                
        elif self.model_type == "roberta":
            try:
                from transformers import pipeline
                # Load pre-trained cardiffnlp/twitter-roberta-base-sentiment-latest
                # (Output mapping: Label 0 -> negative, Label 1 -> neutral, Label 2 -> positive)
                model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
                print(f"Loading HF Transformer Model '{model_name}' on CPU...")
                self.hf_pipeline = pipeline(
                    "sentiment-analysis", 
                    model=model_name, 
                    tokenizer=model_name,
                    device=-1 # Forced to CPU to prevent CUDA memory overflows on dev environments
                )
                print("Loaded Transformer Model: Twitter-RoBERTa.")
            except ImportError:
                print("Error: 'transformers' library not found. Run pip install transformers.")
                print("Falling back to VADER...")
                self.model_type = "vader"
                self._load_model()
            except Exception as e:
                print(f"Error loading RoBERTa: {e}. Falling back to VADER...")
                self.model_type = "vader"
                self._load_model()

    def predict(self, text):
        """Standardized prediction function.
        
        Returns:
            dict: {"label": "Positive"/"Negative"/"Neutral", "score": float}
        """
        if not text or not text.strip():
            return {"label": "Neutral", "score": 0.0}
            
        if self.model_type == "vader":
            # Clean text slightly but retain punctuation (VADER reads punctuation context)
            clean_text = self.preprocessor.clean_text(text)
            scores = self.vader_analyzer.polarity_scores(clean_text)
            compound = scores["compound"]
            
            if compound >= 0.05:
                label = "Positive"
            elif compound <= -0.05:
                label = "Negative"
            else:
                label = "Neutral"
                
            # Compound is in range [-1.0, 1.0]. Normalize to a absolute probability index.
            confidence = abs(compound) if label != "Neutral" else 1.0 - (abs(scores["pos"]) + abs(scores["neg"]))
            return {"label": label, "score": round(float(confidence), 4)}
            
        elif self.model_type == "tfidf_logistic":
            # Full NLP Preprocessing (casing, tokenizing, lemmatizing)
            processed_str = self.preprocessor.preprocess_as_string(text)
            
            # Fallback if preprocessing strips everything (e.g. text is just a URL "@handle http://url.com")
            if not processed_str:
                return {"label": "Neutral", "score": 0.5}
                
            # Vectorize
            vec_text = self.vectorizer.transform([processed_str])
            
            # Predict Label & Confidence
            label = self.classifier.predict(vec_text)[0]
            probs = self.classifier.predict_proba(vec_text)[0]
            classes = self.classifier.classes_
            
            # Find probability index matching the predicted class
            class_idx = list(classes).index(label)
            confidence = probs[class_idx]
            
            return {"label": label, "score": round(float(confidence), 4)}
            
        elif self.model_type == "roberta":
            # CardiffNLP model expects raw tweets, no advanced cleaning needed (transformer self-attention reads grammar)
            clean_text = self.preprocessor.clean_text(text)
            # Clip text if it exceeds maximum context length of 512 tokens
            clean_text = clean_text[:1000] 
            
            res = self.hf_pipeline(clean_text)[0]
            hf_label = res["label"].lower() # positive, negative, or neutral
            confidence = res["score"]
            
            # Map labels
            if "positive" in hf_label:
                label = "Positive"
            elif "negative" in hf_label:
                label = "Negative"
            else:
                label = "Neutral"
                
            return {"label": label, "score": round(float(confidence), 4)}
            
        return {"label": "Neutral", "score": 0.0}
