# Social Media Sentiment Analysis: Placement & Interview Prep Guide

This document is designed to help you explain this project in interviews, summarize it on your resume, and understand the technical trade-offs.

---

## 📄 Resume Bullet Points (Copy & Paste)

Here are three high-impact descriptions of this project for your resume:

*   **Social Media Sentiment Analysis Pipeline & Interactive Dashboard**
    *   Designed and built a modular real-time sentiment tracking dashboard using **Python**, **Streamlit**, and **Plotly** to monitor brand reputation across simulated social channels.
    *   Implemented an end-to-end data pipeline: cleaned noisy social posts using **Regex** filters; tokenized and lemmatized texts via **NLTK**; and persisted results in a local **SQLite** database optimized with indexes on timestamps and topics.
    *   Trained and serialized a custom **TF-IDF + Logistic Regression** text classifier on 100,000 tweets, achieving a balanced F1-score of ~80%, and integrated it alongside **NLTK VADER** and **Hugging Face RoBERTa** Transformer models.

---

## 💬 Technical Interview Q&A (Be Prepared to Answer)

### Q1: Why did you use SQLite instead of a heavy database like PostgreSQL or MongoDB?
**Answer:** 
"For local prototyping and portfolio validation, SQLite is the ideal choice because it is a serverless, file-based database. It requires zero configuration or installations for recruiters running the project from GitHub, which removes setup friction. However, in a production environment, SQLite would suffer from locking issues during concurrent write streams. In production, I would swap SQLite for **PostgreSQL** or a time-series database like **InfluxDB**, coupled with a message queue like **Apache Kafka** to handle ingestion spike loads."

### Q2: What is the difference between CountVectorizer (Bag-of-Words) and TF-IDF?
**Answer:**
"**CountVectorizer** simply counts the absolute frequency of words in each document. This means common, uninformative words like 'is', 'the', and 'a' get high weights. 
**TF-IDF (Term Frequency-Inverse Document Frequency)** solves this by multiplying term frequency by its inverse document frequency. If a word appears in almost *every* tweet in the dataset (like 'phone' in a tech dataset), its TF-IDF weight decreases. If a word appears frequently in only a few documents (like 'defective' or 'outstanding'), it receives a higher weight, allowing the classifier to lock onto strong sentiment indicators."

### Q3: Why did you lemmatize your tokens instead of stemming them?
**Answer:**
"**Stemming** is a crude heuristic process that chops off the ends of words using fixed algorithms (like the Porter Stemmer). For example, 'studies' and 'studying' might reduce to 'studi', which is not a real dictionary word.
**Lemmatization** uses a vocabulary database (like WordNet) and morphological analysis to return the actual dictionary root form, or *lemma*. For example, 'studies' becomes 'study', and 'better' becomes 'good'. Lemmatization preserves grammatical and semantic meaning, though it is slightly slower than stemming."

### Q4: Why does VADER perform well on social media text without training?
**Answer:**
"**VADER (Valence Aware Dictionary and sEntiment Reasoner)** is a lexicon and rule-based sentiment analysis tool specifically tuned to social media micro-blogging style. It doesn't just look at word meanings; it understands social media grammatical nuances:
*   **Capitalization:** 'GREAT' is scored as more intense than 'great'.
*   **Punctuation:** 'Good!!!' gets a higher sentiment intensity than 'Good'.
*   **Emojis:** It scores emoticons (e.g., `:)`, `😡`, `😍`) directly.
*   **Degree modifiers:** 'Extremely bad' amplifies the negative score, while 'hardly bad' diminishes it."

### Q5: How would you scale this system to handle a production load of 10,000 tweets per second?
**Answer:**
"To scale this system to enterprise levels, I would redesign the architecture:
1.  **Ingestion:** Use **Apache Kafka** or **AWS Kinesis** as a distributed message broker to queue incoming raw social media feeds.
2.  **Processing Nodes:** Package the sentiment engine as a containerized microservice using **Docker** and deploy it to a **Kubernetes** cluster for horizontal auto-scaling.
3.  **Model Inference:** Move the model weights out of the app script and serve them via a dedicated model server like **Triton Model Analyzer** or **TensorFlow Serving**, utilizing GPU instances to run Transformer batch inferences.
4.  **Frontend:** Decouple Streamlit and build a production dashboard using **React.js** querying a read-optimized data warehouse like **ClickHouse** or **Elasticsearch**."

---

## 🌐 How to Deploy this Project (Showcase it Live!)

To show recruiters a live working version of this project, you can host it on the free **Streamlit Community Cloud**:

1.  **Push your code to GitHub:** Ensure your folder is committed (excluding `.venv/` and `data/database.sqlite`).
2.  **Sign up for Streamlit Cloud:** Go to [share.streamlit.io](https://share.streamlit.io) and log in with your GitHub account.
3.  **Deploy a New App:**
    *   Select your Repository: `Social-Media-Sentiment-Analysis-Dashboard`.
    *   Select the Branch: `main`.
    *   Select the Main File Path: `app/dashboard.py`.
4.  **Set Environment Variables (if needed):** Under advanced settings, you can define configurations.
5.  **Run the model trainer:** Since model weights (`.pkl` files) are not pushed to Git, add a fallback trigger in `app/dashboard.py` (which is already implemented in `SentimentEngine`) so that it trains the classifier on first load using `data/raw_tweets.csv` if pickle weights are missing, or push a compressed model weight file to git if under 50MB.
