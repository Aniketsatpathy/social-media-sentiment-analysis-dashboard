import os
import sys
import pytest

# Adjust path to import from src folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from preprocessor import Preprocessor

@pytest.fixture
def preprocessor():
    return Preprocessor()

def test_clean_text_urls(preprocessor):
    text = "Visit http://example.com or https://google.com/test for details."
    cleaned = preprocessor.clean_text(text)
    assert "http" not in cleaned
    assert "https" not in cleaned
    assert "example.com" not in cleaned
    assert cleaned == "Visit or for details."

def test_clean_text_mentions(preprocessor):
    text = "Hey @elonmusk, check out the new design from @Apple!"
    cleaned = preprocessor.clean_text(text)
    assert "@elonmusk" not in cleaned
    assert "@Apple" not in cleaned
    assert cleaned == "Hey , check out the new design from !"

def test_clean_text_html_entities(preprocessor):
    text = "Coffee &amp; coding &lt; sleeping."
    cleaned = preprocessor.clean_text(text)
    assert "&amp;" not in cleaned
    assert "&lt;" not in cleaned
    # Single spaces standardized
    assert "Coffee" in cleaned
    assert "coding" in cleaned

def test_preprocess_lemmatization(preprocessor):
    tokens = preprocessor.preprocess("The batteries are defective")
    # 'batteries' -> 'battery'
    assert "battery" in tokens
    assert "defective" in tokens
    # Stopwords like 'the', 'are' removed
    assert "the" not in tokens
    assert "are" not in tokens

def test_preprocess_negations_retained(preprocessor):
    tokens = preprocessor.preprocess("This screen is not good")
    # Negations like 'not' are kept by preprocessor to avoid losing negative context
    assert "not" in tokens
    assert "good" in tokens
