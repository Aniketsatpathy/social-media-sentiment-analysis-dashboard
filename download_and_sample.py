import os
import urllib.request
import zipfile
import csv
import random

def download_and_sample():
    url = "http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip"
    zip_path = "trainingandtestdata.zip"
    data_dir = "data"
    output_path = os.path.join(data_dir, "raw_tweets.csv")
    csv_filename = "training.1600000.processed.noemoticon.csv"
    
    # 1. Create data directory if it doesn't exist
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created directory: {data_dir}")
        
    # 2. Download ZIP
    if not os.path.exists(zip_path):
        print(f"Downloading Sentiment140 dataset from {url}...")
        print("This zip file is ~77MB, please wait...")
        urllib.request.urlretrieve(url, zip_path)
        print("Download complete.")
    else:
        print("ZIP file already exists.")

    # 3. Extract the target CSV
    print(f"Extracting {csv_filename} from zip...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extract(csv_filename, ".")
    print("Extraction complete.")

    # 4. Parse and subset the data
    print("Sampling 50,000 positive and 50,000 negative tweets...")
    negatives = []
    positives = []
    
    # Read the full dataset (using ISO-8859-1 encoding since social media tweets contain special characters)
    with open(csv_filename, mode='r', encoding='ISO-8859-1') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 6:
                continue
            label = row[0] # '0' = negative, '4' = positive
            text = row[5]
            user = row[4]
            date = row[2]
            tweet_id = row[1]
            query = row[3]
            
            record = [label, tweet_id, date, query, user, text]
            
            if label == '0':
                negatives.append(record)
            elif label == '4':
                positives.append(record)
                
    print(f"Found {len(negatives)} negative tweets and {len(positives)} positive tweets.")
    
    # Sample balanced set
    sampled_negatives = random.sample(negatives, min(50000, len(negatives)))
    sampled_positives = random.sample(positives, min(50000, len(positives)))
    
    combined = sampled_negatives + sampled_positives
    random.shuffle(combined)
    
    # 5. Write to output CSV
    print(f"Writing sampled data to {output_path}...")
    headers = ["sentiment", "id", "date", "query", "user", "text"]
    with open(output_path, mode='w', encoding='utf-8', newline='') as out_f:
        writer = csv.writer(out_f)
        writer.writerow(headers)
        writer.writerows(combined)
        
    # 6. Clean up temporary files to save disk space
    print("Cleaning up temporary zip and large extracted files...")
    if os.path.exists(zip_path):
        os.remove(zip_path)
    if os.path.exists(csv_filename):
        os.remove(csv_filename)
        
    print(f"Data validation: Check if file exists. {output_path} generated successfully with {len(combined)} rows.")

if __name__ == "__main__":
    download_and_sample()
