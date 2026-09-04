import pandas as pd
import numpy as np

class DataValidator:
    """Validator utility to check structure, schema, quality, and distribution of dataset files."""
    
    @staticmethod
    def validate_csv_schema(file_path, required_cols):
        """Validates that a CSV file exists and contains the necessary schema columns."""
        try:
            # Read only the header to speed up validation on large datasets
            df_head = pd.read_csv(file_path, nrows=0, encoding='ISO-8859-1')
            columns = set(df_head.columns)
            missing = [col for col in required_cols if col not in columns]
            if missing:
                return False, f"Missing required columns: {missing}"
            return True, "Schema is valid."
        except FileNotFoundError:
            return False, f"File not found: {file_path}"
        except Exception as e:
            return False, f"Error reading CSV header: {str(e)}"

    @staticmethod
    def profile_dataset(df, text_col="text", label_col="sentiment"):
        """Performs data quality profiling on a pandas DataFrame."""
        profile = {}
        
        # Row counts
        total_rows = len(df)
        profile["total_records"] = total_rows
        
        if total_rows == 0:
            return {"error": "DataFrame is empty"}

        # Null checks
        profile["null_text_count"] = int(df[text_col].isnull().sum())
        profile["empty_text_count"] = int((df[text_col].astype(str).str.strip() == "").sum())
        
        # Duplicates check
        profile["duplicate_records"] = int(df.duplicated(subset=[text_col]).sum())
        profile["duplicate_percentage"] = round((profile["duplicate_records"] / total_rows) * 100, 2)
        
        # Character bounds
        text_lengths = df[text_col].dropna().astype(str).str.len()
        profile["char_length_stats"] = {
            "min": int(text_lengths.min()) if not text_lengths.empty else 0,
            "max": int(text_lengths.max()) if not text_lengths.empty else 0,
            "mean": round(text_lengths.mean(), 1) if not text_lengths.empty else 0.0
        }
        
        # Word counts bounds
        word_counts = df[text_col].dropna().astype(str).str.split().apply(len)
        profile["word_count_stats"] = {
            "min": int(word_counts.min()) if not word_counts.empty else 0,
            "max": int(word_counts.max()) if not word_counts.empty else 0,
            "mean": round(word_counts.mean(), 1) if not word_counts.empty else 0.0
        }
        
        # Label distributions
        if label_col in df.columns:
            counts = df[label_col].value_counts()
            dist = {}
            for label, count in counts.items():
                dist[str(label)] = {
                    "count": int(count),
                    "percentage": round((count / total_rows) * 100, 2)
                }
            profile["label_distribution"] = dist
            
        return profile

    @staticmethod
    def check_class_balance(df, label_col="sentiment", skew_threshold=80.0):
        """Verifies if the class distribution is heavily skewed, issuing a warning flag."""
        if label_col not in df.columns:
            return True, "No label column available to check balance."
            
        total = len(df)
        if total == 0:
            return False, "Dataset is empty."
            
        counts = df[label_col].value_counts()
        for label, count in counts.items():
            pct = (count / total) * 100
            if pct > skew_threshold:
                return False, f"Severe label imbalance detected! Class '{label}' accounts for {pct:.1f}% of the dataset."
                
        return True, "Labels are balanced."

    @staticmethod
    def detect_non_ascii_outliers(df, text_col="text", threshold=0.1):
        """Detects the ratio of tweets containing non-ascii text elements (e.g. non-English scripts)."""
        total = len(df)
        if total == 0:
            return 0.0
            
        def is_ascii(s):
            try:
                s.encode('ascii')
                return True
            except UnicodeEncodeError:
                return False
                
        non_ascii_count = df[text_col].dropna().apply(lambda x: not is_ascii(str(x))).sum()
        ratio = non_ascii_count / total
        return round(ratio * 100, 2)
