import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import re
from nltk.corpus import stopwords
import joblib

def load_and_preprocess_data():
    """Time to load up our fake vs real news datasets!"""
    # Load datasets
    url_fake = "D:\\Kartik\\Learning\\ML\\Data\\Fake.csv"
    url_true = "D:\\Kartik\\Learning\\ML\\Data\\True.csv"
    df_fake = pd.read_csv(url_fake)
    df_true = pd.read_csv(url_true)
    
    # Add class labels (0 = fake, 1 = real - pretty straightforward!)
    df_fake["class"] = 0  # Fake news
    df_true["class"] = 1  # Real news
    
    # Combine datasets (mix them all together)
    df = pd.concat([df_fake, df_true], axis=0)
    
    # Shuffle the data (we don't want all fake news at the top!)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Combine title and text (the more text, the better our AI can learn)
    df["total_text"] = df["title"] + " " + df["text"]
    
    return df

def clean_text(df):
    """Let's clean up this messy text data - AI likes things neat and tidy!"""
    # Convert to lowercase (because "NEWS" and "news" should be treated the same)
    df["total_text"] = df["total_text"].str.lower()
    
    # Drop unnecessary columns (we don't need them anymore)
    df = df.drop(["title", "text", "date"], axis=1)
    
    # Remove special characters and keep only letters and spaces (no weird symbols!)
    df["total_text"] = df["total_text"].apply(lambda x: re.sub(r'[^a-z\s]', '', x))
    
    # Remove stopwords (common words like "the", "and", "is" don't help much)
    stop_words = set(stopwords.words("english"))
    df["total_text"] = df["total_text"].apply(
        lambda x: " ".join([word for word in x.split() if word not in stop_words])
    )
    return df

def create_features(df):
    """Turn our clean text into numbers that the AI can actually understand"""
    vectorizer = TfidfVectorizer(max_features=5000)  # Keep the top 5000 most important words
    X = vectorizer.fit_transform(df["total_text"])
    y = df["class"]
    
    return X, y, vectorizer

def train_svm_model(X, y):
    """Time to train our fake news detective! This is where the real learning happens."""
    # Split data (80% for training, 20% for testing - gotta keep some data to test with!)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42, test_size=0.2
    )
    
    # Train SVM model (this is our main AI brain!)
    model = SVC(kernel="linear")  # Linear kernel works great for text classification
    model.fit(X_train, y_train)
    
    # Make predictions on our test data (let's see how we did!)
    y_pred = model.predict(X_test)
    
    return model, X_test, y_test, y_pred

def evaluate_model(y_test, y_pred):
    """Let's see how well our AI detective performed!"""
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f} (that's {accuracy*100:.2f}% - not bad!)")
    print("\nDetailed Performance Report:")
    print(classification_report(y_test, y_pred))
    
    # Create a nice confusion matrix visualization
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
    plt.title('How Well Did Our Detective Do?')
    plt.ylabel('What It Actually Was')
    plt.xlabel('What Our AI Predicted')
    plt.show()
    
    return accuracy

def save_model(model, vectorizer):
    """Save our trained AI so we don't have to train it again later"""
    joblib.dump(model, "svm_model.pkl")
    joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
    print("Model and vectorizer saved successfully! They're ready to use.")

def main():
    """The main show! Let's train our fake news detective from start to finish."""
    print("Starting the fake news detective training program!")
    print("Loading and preprocessing data...")
    df = load_and_preprocess_data()
    
    print("Cleaning up the text data...")
    df = clean_text(df)
    
    print("Converting text to numbers (TF-IDF magic)...")
    X, y, vectorizer = create_features(df)
    
    print("Training our SVM detective (this is where the AI learns)...")
    model, X_test, y_test, y_pred = train_svm_model(X, y)
    
    print("Evaluating how well our detective performed...")
    accuracy = evaluate_model(y_test, y_pred)
    
    print("Saving our trained detective for future use...")
    save_model(model, vectorizer)
    
    print(f"\nTraining completed! Our fake news detective achieved {accuracy:.4f} accuracy!")
    print("Ready to catch some fake news! Run 'python app.py' to try it out.")

if __name__ == "__main__":
    main()
