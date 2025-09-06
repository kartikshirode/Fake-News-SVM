"""
Enhanced Fake News Detection Training Script
Focuses on accuracy and robustness with a single comprehensive script
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
import time
import joblib
import warnings

# Try to import nltk resources, with graceful fallback
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    
    # Check if stopwords are available, download if needed
    try:
        stopwords.words('english')
    except LookupError:
        print("Downloading NLTK stopwords...")
        nltk.download('stopwords', quiet=True)
    
    # Check if wordnet is available, download if needed
    try:
        WordNetLemmatizer()
        try:
            # Try to actually use it to make sure it's fully loaded
            lemmatizer = WordNetLemmatizer()
            lemmatizer.lemmatize("test")
        except LookupError:
            print("Downloading NLTK WordNet...")
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)
    except Exception as e:
        print(f"Warning: WordNet not available: {str(e)}")
        print("Will use simpler text processing")
        NLTK_AVAILABLE = False
        
    NLTK_AVAILABLE = True
except ImportError:
    print("NLTK not available. Will use basic text processing.")
    NLTK_AVAILABLE = False

warnings.filterwarnings('ignore')

def find_data_files():
    """Find the data files in various possible locations"""
    possible_fake_paths = [
        "D:\\Kartik\\Learning\\ML\\Data\\Fake.csv",
        "..\\Data\\Fake.csv",
        "..\\..\\Data\\Fake.csv",
        "Data\\Fake.csv",
        "Fake.csv"
    ]
    
    possible_true_paths = [
        "D:\\Kartik\\Learning\\ML\\Data\\True.csv",
        "..\\Data\\True.csv",
        "..\\..\\Data\\True.csv",
        "Data\\True.csv",
        "True.csv"
    ]
    
    fake_path = None
    true_path = None
    
    for path in possible_fake_paths:
        if os.path.exists(path):
            fake_path = path
            break
    
    for path in possible_true_paths:
        if os.path.exists(path):
            true_path = path
            break
    
    if fake_path and true_path:
        print(f"Found data files:\n- Fake news: {fake_path}\n- Real news: {true_path}")
        return fake_path, true_path
    else:
        raise FileNotFoundError("Could not find Fake.csv and True.csv files in any expected location")

def load_and_preprocess_data(max_samples=None):
    """Load datasets with better sampling to reduce bias"""
    # Find and load datasets
    url_fake, url_true = find_data_files()
    df_fake = pd.read_csv(url_fake)
    df_true = pd.read_csv(url_true)
    
    print(f"Original data: Fake={len(df_fake)}, Real={len(df_true)}")
    
    # Balance the dataset by taking equal samples
    min_samples = min(len(df_fake), len(df_true))
    if max_samples:
        sample_size = min(min_samples, max_samples)
    else:
        sample_size = min_samples  # Use all available data if no limit specified
    
    df_fake_sampled = df_fake.sample(n=sample_size, random_state=42)
    df_true_sampled = df_true.sample(n=sample_size, random_state=42)
    
    print(f"Balanced data: Fake={len(df_fake_sampled)}, Real={len(df_true_sampled)}")
    
    # Add class labels
    df_fake_sampled["class"] = 0  # Fake news
    df_true_sampled["class"] = 1  # Real news
    
    # Combine datasets
    df = pd.concat([df_fake_sampled, df_true_sampled], axis=0)
    
    # Shuffle the data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df

def clean_text_advanced(text):
    """Advanced text cleaning to preserve important meaning while removing noise"""
    if pd.isna(text):
        return ""
    
    # Convert to string and lowercase
    text = str(text).lower()
    
    # Replace URLs with placeholder
    text = re.sub(r'http\S+|www\S+|https\S+', ' [URL] ', text, flags=re.MULTILINE)
    
    # Replace email with placeholder
    text = re.sub(r'\S+@\S+', ' [EMAIL] ', text)
    
    # Replace numbers with placeholder but keep important context
    text = re.sub(r'\b\d{4,}\b', ' [NUMBER] ', text)  # Replace long numbers
    
    # Keep important punctuation that may indicate sentiment or emphasis
    text = re.sub(r'[^a-zA-Z0-9\s\.\!\?\,\;\:\"\'\(\)\-]', ' ', text)
    
    # Replace multiple punctuation with single instance (e.g., !!! → !)
    text = re.sub(r'([\.\!\?\,\;\:\"\'\(\)\-])\1+', r'\1', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Advanced processing if NLTK is available
    if NLTK_AVAILABLE:
        try:
            # More sophisticated stopword removal
            stop_words = set(stopwords.words("english"))
            # Keep important words that might indicate bias, sentiment, or opinion
            keep_words = {
                'not', 'no', 'never', 'nothing', 'nobody', 'neither', 'nowhere', 'none',
                'but', 'however', 'although', 'though', 'yet', 'still', 'nevertheless',
                'very', 'really', 'quite', 'extremely', 'highly', 'completely',
                'must', 'should', 'would', 'could', 'might', 'may',
                'always', 'never', 'often', 'sometimes', 'usually',
                'true', 'false', 'fact', 'actual', 'allegedly', 'reportedly'
            }
            stop_words = stop_words - keep_words
            
            # Tokenize and apply lemmatization for better feature extraction
            lemmatizer = WordNetLemmatizer()
            words = text.split()
            processed_words = []
            
            for word in words:
                if word not in stop_words and len(word) > 1:
                    try:
                        # Apply lemmatization to reduce words to their base form
                        lemmatized = lemmatizer.lemmatize(word)
                        processed_words.append(lemmatized)
                    except:
                        # If lemmatization fails, just use the original word
                        processed_words.append(word)
                    
            text = ' '.join(processed_words)
        except Exception as e:
            # If any NLTK processing fails, fall back to simpler processing
            print(f"Warning: NLTK processing failed: {str(e)}")
            print("Using simpler text processing")
            words = text.split()
            text = ' '.join([word for word in words if len(word) > 1])
    else:
        # Simpler processing if NLTK is not available
        words = text.split()
        text = ' '.join([word for word in words if len(word) > 1])
    
    return text

def preprocess_dataframe(df):
    """Process dataframe with better handling"""
    start_time = time.time()
    print("Preprocessing text data...")
    
    # Combine title and text
    if 'title' in df.columns and 'text' in df.columns:
        df["total_text"] = df["title"].fillna("").astype(str) + " " + df["text"].fillna("").astype(str)
    elif 'text' in df.columns:
        df["total_text"] = df["text"].fillna("").astype(str)
    else:
        # Try to find text columns with different names
        text_columns = [col for col in df.columns if col.lower() in ['text', 'content', 'article']]
        title_columns = [col for col in df.columns if col.lower() in ['title', 'headline']]
        
        if text_columns and title_columns:
            df["total_text"] = df[title_columns[0]].fillna("").astype(str) + " " + df[text_columns[0]].fillna("").astype(str)
        elif text_columns:
            df["total_text"] = df[text_columns[0]].fillna("").astype(str)
        else:
            raise ValueError("Could not find text columns in the dataset")
    
    # Clean text (use advanced cleaning)
    print("Applying text cleaning...")
    df["total_text"] = df["total_text"].apply(clean_text_advanced)
    
    # Remove very short or empty texts
    df = df[df["total_text"].str.len() > 30]
    
    # Remove exact duplicates
    df = df.drop_duplicates(subset=['total_text'])
    
    # Print text statistics
    text_lengths = df["total_text"].str.len()
    print(f"Text length statistics:")
    print(f"- Average: {text_lengths.mean():.1f} characters")
    print(f"- Minimum: {text_lengths.min()} characters")
    print(f"- Maximum: {text_lengths.max()} characters")
    
    print(f"After preprocessing: {len(df)} articles remaining")
    print(f"Preprocessing completed in {time.time() - start_time:.2f} seconds")
    
    return df[["total_text", "class"]]

def create_optimized_features(df, max_features=3000):
    """Create features with optimized parameters for better accuracy"""
    start_time = time.time()
    print(f"Creating TF-IDF features with {max_features} features...")
    
    # Advanced TF-IDF settings
    vectorizer = TfidfVectorizer(
        max_features=max_features,  # More features for better accuracy
        min_df=3,                  # Must appear in at least 3 documents
        max_df=0.9,                # Can appear in up to 90% of documents
        ngram_range=(1, 2),        # Unigrams and bigrams
        strip_accents='unicode',
        lowercase=True,
        sublinear_tf=True,         # Use sublinear TF scaling
        use_idf=True,
        norm='l2'                  # L2 normalization
    )
    
    X = vectorizer.fit_transform(df["total_text"])
    y = df["class"]
    
    print(f"Feature matrix shape: ({X.shape[0]}, {X.shape[1]})")
    print(f"Class distribution: Real={sum(y)}, Fake={len(y) - sum(y)}")
    print(f"Feature extraction completed in {time.time() - start_time:.2f} seconds")
    
    return X, y, vectorizer

def train_optimized_model(X, y):
    """Train a highly optimized model for accuracy"""
    start_time = time.time()
    print("Training optimal model...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training data: {X_train.shape[0]} samples")
    print(f"Testing data: {X_test.shape[0]} samples")
    
    # Define models with optimized parameters
    svm = SVC(
        kernel='linear',
        C=1.0,
        class_weight='balanced',
        random_state=42,
        probability=True
    )
    
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight='balanced',
        random_state=42
    )
    
    lr = LogisticRegression(
        C=1.0,
        solver='liblinear',
        class_weight='balanced',
        random_state=42,
        max_iter=1000
    )
    
    # Create a voting ensemble
    print("Using ensemble voting for better accuracy...")
    ensemble = VotingClassifier(
        estimators=[
            ('svm', svm),
            ('rf', rf),
            ('lr', lr)
        ],
        voting='soft'  # Use probability weights
    )
    
    # Train the ensemble
    ensemble.fit(X_train, y_train)
    
    # Predict on test set
    y_pred = ensemble.predict(X_test)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    
    print(f"Training completed in {time.time() - start_time:.2f} seconds")
    print(f"Ensemble model accuracy: {accuracy:.4f}")
    print(f"Ensemble model F1-score: {f1:.4f}")
    
    return ensemble, X_test, y_test, y_pred

def evaluate_model_detailed(y_test, y_pred, model_name="Model"):
    """Detailed evaluation focusing on accuracy and balance"""
    print("\nDetailed Model Evaluation:")
    print("-" * 50)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    
    print(f"Overall Performance:")
    print(f"- Accuracy: {accuracy:.4f}")
    print(f"- F1-Score: {f1:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Fake News', 'Real News']))
    
    # Confusion matrix analysis
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print("\nConfusion Matrix:")
    print(f"- True Negatives (correctly identified FAKE): {tn}")
    print(f"- False Positives (FAKE misclassified as REAL): {fp}")
    print(f"- False Negatives (REAL misclassified as FAKE): {fn}")
    print(f"- True Positives (correctly identified REAL): {tp}")
    
    # Calculate balance metrics
    fake_precision = tn / (tn + fn) if (tn + fn) > 0 else 0
    real_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    fake_recall = tn / (tn + fp) if (tn + fp) > 0 else 0
    real_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    print("\nBalance Analysis:")
    print(f"- Fake News - Precision: {fake_precision:.4f}, Recall: {fake_recall:.4f}")
    print(f"- Real News - Precision: {real_precision:.4f}, Recall: {real_recall:.4f}")
    
    # Check for bias
    bias_towards_fake = fn / (fn + tp) if (fn + tp) > 0 else 0
    bias_towards_real = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    print("\nBias Analysis:")
    print(f"- Tendency to predict FAKE: {bias_towards_fake:.4f}")
    print(f"- Tendency to predict REAL: {bias_towards_real:.4f}")
    
    if bias_towards_fake > 0.1 or bias_towards_real > 0.1:
        if bias_towards_fake > bias_towards_real:
            print("- Model shows some bias towards predicting FAKE news")
        else:
            print("- Model shows some bias towards predicting REAL news")
    else:
        print("- Model is well balanced!")
    
    return accuracy, f1

def test_real_examples(model, vectorizer):
    """Test the model on real-world examples"""
    print("\nTesting on real-world examples:")
    print("-" * 50)
    
    examples = [
        {
            "text": "Scientists discover new species of frog in Amazon rainforest with unique coloration.",
            "expected": "Real"
        },
        {
            "text": "Breaking news: Scientists discover that drinking coffee mixed with lemon juice can cure all types of cancer overnight.",
            "expected": "Fake"
        },
        {
            "text": "The Federal Reserve announced a quarter-point interest rate increase today, bringing the target range to 5.25-5.5%.",
            "expected": "Real"
        },
        {
            "text": "Secret government documents reveal that aliens have been living among us for decades and have infiltrated the highest levels of government.",
            "expected": "Fake"
        },
        {
            "text": "The annual budget meeting is scheduled for next Tuesday at 2:00 PM in Conference Room A. All department heads are required to attend.",
            "expected": "Real"
        }
    ]
    
    correct_count = 0
    
    for i, example in enumerate(examples, 1):
        # Preprocess text
        processed_text = clean_text_advanced(example["text"])
        
        # Vectorize
        text_vector = vectorizer.transform([processed_text])
        
        # Predict
        prediction = model.predict(text_vector)[0]
        
        # Calculate confidence
        confidence = 0.5
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(text_vector)[0]
            confidence = probs[1] if prediction == 1 else probs[0]
        
        # Format result
        result = "REAL" if prediction == 1 else "FAKE"
        is_correct = (result == "REAL" and example["expected"] == "Real") or \
                     (result == "FAKE" and example["expected"] == "Fake")
        
        if is_correct:
            correct_count += 1
        
        # Print result
        print(f"Example {i}:")
        print(f"Text: {example['text']}")
        print(f"Expected: {example['expected']}")
        print(f"Predicted: {result} with {confidence:.2f} confidence")
        print(f"Correct: {'Yes' if is_correct else 'No'}")
        print("-" * 50)
    
    # Print summary
    accuracy = correct_count / len(examples)
    print(f"Real-world test accuracy: {accuracy:.2f} ({correct_count}/{len(examples)} correct)")
    
    return accuracy

def save_model(model, vectorizer):
    """Save the model and vectorizer"""
    print("\nSaving model and vectorizer...")
    
    try:
        joblib.dump(model, "svm_model.pkl")
        joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
        print("Model and vectorizer saved successfully!")
    except Exception as e:
        print(f"Error saving model: {str(e)}")

def main():
    """Main function to train and evaluate the fake news detection model"""
    total_start_time = time.time()
    
    print("=" * 60)
    print("ENHANCED FAKE NEWS DETECTION - TRAINING")
    print("=" * 60)
    
    # Step 1: Load and preprocess data
    df = load_and_preprocess_data()
    
    # Step 2: Preprocess text
    df = preprocess_dataframe(df)
    
    # Step 3: Create features
    X, y, vectorizer = create_optimized_features(df)
    
    # Step 4: Train model
    model, X_test, y_test, y_pred = train_optimized_model(X, y)
    
    # Step 5: Evaluate model
    accuracy, f1 = evaluate_model_detailed(y_test, y_pred, "Ensemble Model")
    
    # Step 6: Test on real examples
    real_world_accuracy = test_real_examples(model, vectorizer)
    
    # Step 7: Save model
    save_model(model, vectorizer)
    
    # Print summary
    total_time = time.time() - total_start_time
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Total training time: {total_time:.2f} seconds")
    print(f"Model accuracy: {accuracy:.4f}")
    print(f"F1 score: {f1:.4f}")
    print(f"Real-world accuracy: {real_world_accuracy:.2f}")
    print("=" * 60)
    print("To test the model, run: python app.py")
    print("=" * 60)

if __name__ == "__main__":
    main()
