import argparse
import sys
import os

# Add src to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from database_manager import DBManager
from data_generator import DataGenerator
from model_trainer import train_model
from data_validator import DataValidator
import pandas as pd

def run_validation():
    csv_path = "data/raw_tweets.csv"
    print(f"=== Running Data Validation Checks on {csv_path} ===")
    required_cols = ["text", "sentiment"]
    
    # Verify Schema
    valid, msg = DataValidator.validate_csv_schema(csv_path, required_cols)
    print(f"1. Schema Check: {'✅ PASSED' if valid else '❌ FAILED'}")
    print(f"   Details: {msg}")
    
    if not valid:
        return
        
    # Read and Profile
    df = pd.read_csv(csv_path)
    profile = DataValidator.profile_dataset(df, "text", "sentiment")
    
    print("\n2. Dataset Quality Profile Summary:")
    print(f"   - Total Rows: {profile.get('total_records')}")
    print(f"   - Null Tweets: {profile.get('null_text_count')}")
    print(f"   - Empty Strings: {profile.get('empty_text_count')}")
    print(f"   - Duplicate Rate: {profile.get('duplicate_percentage')}%")
    
    # Class Balance
    balanced, balance_msg = DataValidator.check_class_balance(df, "sentiment")
    print(f"\n3. Class Balance: {'✅ BALANCED' if balanced else '⚠️ IMBALANCED'}")
    print(f"   Details: {balance_msg}")
    
    # Non-English Outliers
    non_ascii = DataValidator.detect_non_ascii_outliers(df, "text")
    print(f"\n4. Non-English / ASCII Outliers Ratio: {non_ascii}%")
    print("\n=== Validation Complete ===")

def main():
    parser = argparse.ArgumentParser(description="Social Media Sentiment Analysis Pipeline Manager")
    
    parser.add_argument("--train", action="store_true", help="Clean dataset and train TF-IDF + Classifier model")
    parser.add_argument("--validate", action="store_true", help="Validate raw dataset csv structure and quality profiles")
    parser.add_argument("--simulate", action="store_true", help="Start the simulated streaming feed writing to SQLite")
    parser.add_argument("--clear-db", action="store_true", help="Wipe all entries in the local SQLite database")
    
    args = parser.parse_args()
    
    # If no flags are passed, print usage and exit
    if not any(vars(args).values()):
        parser.print_help()
        print("\nExample commands:")
        print("  python main.py --validate")
        print("  python main.py --train")
        print("  python main.py --simulate")
        sys.exit(0)
        
    if args.validate:
        run_validation()
        
    if args.train:
        train_model()
        
    if args.simulate:
        generator = DataGenerator()
        delay = generator.config["simulation"].get("delay_seconds", 1.5)
        generator.start_simulation(delay=delay, batch_size=2)
        
    if args.clear_db:
        db = DBManager()
        db.clear_database()

if __name__ == "__main__":
    main()
