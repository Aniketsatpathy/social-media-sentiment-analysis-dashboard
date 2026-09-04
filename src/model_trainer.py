import os
import pickle
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from preprocessor import Preprocessor
from data_validator import DataValidator

def train_model(config_path="config/config.yaml"):
    # Load configuration
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
        
    vec_path = config["model_paths"]["vectorizer"]
    clf_path = config["model_paths"]["classifier"]
    
    # Ensure parent folders exist
    os.makedirs(os.path.dirname(vec_path), exist_ok=True)
    os.makedirs(os.path.dirname(clf_path), exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    
    csv_path = "data/raw_tweets.csv"
    
    # 1. Validate dataset schema
    required_cols = ["text", "sentiment"]
    valid, msg = DataValidator.validate_csv_schema(csv_path, required_cols)
    if not valid:
        print(f"Error during dataset schema check: {msg}")
        return False
        
    # 2. Load dataset
    print(f"Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path, encoding='utf-8')
    
    # 3. Profile dataset
    profile = DataValidator.profile_dataset(df, "text", "sentiment")
    print(f"Dataset Profiling Results:")
    print(f"  - Total Records: {profile.get('total_records')}")
    print(f"  - Duplicate Records: {profile.get('duplicate_records')} ({profile.get('duplicate_percentage')}%)")
    print(f"  - Char lengths (mean): {profile.get('char_length_stats', {}).get('mean')} chars")
    
    # Check class balance
    balanced, balance_msg = DataValidator.check_class_balance(df, "sentiment")
    print(f"  - Class Balance Status: {balance_msg}")
    
    # Check non-English noise
    non_ascii_pct = DataValidator.detect_non_ascii_outliers(df, "text")
    print(f"  - Non-ASCII tweets percentage: {non_ascii_pct}%")

    # Clean missing texts if any
    df = df.dropna(subset=["text"])
    
    # 4. Run Preprocessing Pipeline
    print("Preprocessing tweets (cleaning, tokenization, lemmatization)...")
    preprocessor = Preprocessor()
    
    # Process text into space-separated string representation for TF-IDF vectorizer
    df["clean_text"] = df["text"].apply(lambda x: preprocessor.preprocess_as_string(x))
    
    # Exclude entries that became completely empty after cleaning
    df = df[df["clean_text"].str.strip() != ""]
    
    # Convert labels from 0 (neg) and 4 (pos) to standard strings if loaded from raw Sentiment140
    # Sentiment140 labels are 0 for negative and 4 for positive
    df["label"] = df["sentiment"].replace({0: "Negative", 4: "Positive", "0": "Negative", "4": "Positive"})
    
    # 5. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"], df["label"], test_size=0.20, random_state=42, stratify=df["label"]
    )
    print(f"Split sizes: Train={len(X_train)}, Test={len(X_test)}")
    
    # 6. Feature Extraction (TF-IDF Vectorization)
    print("Extracting TF-IDF Features...")
    vectorizer = TfidfVectorizer(
        max_features=25000,
        ngram_range=(1, 2),
        min_df=2
    )
    
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # 7. Model Training (Logistic Regression)
    print("Training Logistic Regression Model...")
    classifier = LogisticRegression(max_iter=1000, solver='lbfgs', C=1.0)
    classifier.fit(X_train_vec, y_train)
    
    # 8. Evaluation
    predictions = classifier.predict(X_test_vec)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)
    
    print(f"Model Training Complete! Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)
    print("Confusion Matrix:")
    print(cm)
    
    # 9. Save Artifacts
    print(f"Saving vectorizer to {vec_path}...")
    with open(vec_path, 'wb') as f:
        pickle.dump(vectorizer, f)
        
    print(f"Saving classifier to {clf_path}...")
    with open(clf_path, 'wb') as f:
        pickle.dump(classifier, f)
        
    # Write to evaluation report text file
    report_path = "outputs/evaluation_report.txt"
    with open(report_path, 'w') as f:
        f.write("=== Sentiment Model Evaluation Report ===\n")
        f.write(f"Date: {pd.Timestamp.now()}\n")
        f.write(f"Source Dataset: {csv_path}\n")
        f.write(f"Model Architecture: TF-IDF + Logistic Regression\n")
        f.write(f"Accuracy: {accuracy:.4f}\n\n")
        f.write("--- Classification Performance ---\n")
        f.write(report)
        f.write("\n--- Confusion Matrix ---\n")
        f.write(f"{cm}\n")
        f.write(f"\nVectorizer Vocabulary Size: {len(vectorizer.vocabulary_)}\n")
        
    print(f"Evaluation report written to {report_path}")
    return True

if __name__ == "__main__":
    train_model()
