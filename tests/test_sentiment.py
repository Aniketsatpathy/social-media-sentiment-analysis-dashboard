import os
import sys
import pytest

# Adjust path to import from src folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from sentiment_engine import SentimentEngine

def test_sentiment_vader_positive():
    # Force use of VADER for this unit test
    engine = SentimentEngine()
    engine.model_type = "vader"
    engine._load_model()
    
    res = engine.predict("This product is absolutely amazing and perfect!")
    assert res["label"] == "Positive"
    assert res["score"] > 0.5

def test_sentiment_vader_negative():
    engine = SentimentEngine()
    engine.model_type = "vader"
    engine._load_model()
    
    res = engine.predict("This is the worst device I have ever owned. It is broken.")
    assert res["label"] == "Negative"
    assert res["score"] > 0.5

def test_sentiment_vader_neutral():
    engine = SentimentEngine()
    engine.model_type = "vader"
    engine._load_model()
    
    res = engine.predict("I am writing code on my computer.")
    assert res["label"] == "Neutral"

def test_empty_string_prediction():
    engine = SentimentEngine()
    res = engine.predict("")
    assert res["label"] == "Neutral"
    assert res["score"] == 0.0
