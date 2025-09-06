import pandas as pd
import re
import string
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# -----------------------
# 1. Load and preprocess
# -----------------------
def clean_text(text):
    """Basic text cleaning: lowercase, remove punctuation/numbers/extra spaces."""
    text = text.lower()
    text = re.sub(r"\d+", " ", text)  # remove numbers
    text = text.translate(str.maketrans("", "", string.punctuation))  # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()
    return text

def load_data():
    # Load Kaggle fake-news dataset (Fake.csv and True.csv)
    df_fake = pd.read_csv("Fake.csv")
    df_true = pd.read_csv("True.csv")

    df_fake["class"] = 0  # Fake = 0
    df_true["class"] = 1  # Real = 1

    df = pd.concat([df_fake, df_true], axis=0).sample(frac=1, random_state=42).reset_index(drop=True)
    df["total_text"] = (df["title"].fillna("") + " " + df["text"].fillna("")).apply(clean_text)

    return df

# -----------------------
# 2. Features (TF-IDF)
# -----------------------
def create_features(df):
    vectorizer = TfidfVectorizer(
        max_features=10000,
        min_df=5,
        max_df=0.8,
        ngram_range=(1, 3),
        sublinear_tf=True,
        strip_accents="unicode",
        lowercase=True
    )
    X = vectorizer.fit_transform(df["total_text"])
    y = df["class"]
    return X, y, vectorizer

# -----------------------
# 3. Train Logistic Regression
# -----------------------
def train_model(X_train, y_train):
    model = LogisticRegression(
        C=2.0,
        solver="liblinear",
        class_weight="balanced",
        max_iter=2000
    )
    model.fit(X_train, y_train)
    return model

# -----------------------
# 4. Evaluate
# -----------------------
def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=3))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# -----------------------
# 5. Real-world test
# -----------------------
def test_examples(model, vectorizer):
    examples = [
        ("Scientists discover new species of frog in Amazon rainforest with unique coloration.", 1),
        ("Breaking news: Scientists discover that drinking coffee mixed with lemon juice can cure all types of cancer overnight.", 0),
        ("The Federal Reserve announced a quarter-point interest rate increase today, bringing the target range to 5.25-5.5%.", 1),
        ("Secret government documents reveal that aliens have been living among us for decades and have infiltrated the highest levels of government.", 0),
        ("The annual budget meeting is scheduled for next Tuesday at 2:00 PM in Conference Room A. All department heads are required to attend.", 1)
    ]

    correct = 0
    for i, (text, expected) in enumerate(examples, 1):
        vec = vectorizer.transform([clean_text(text)])
        pred = model.predict(vec)[0]
        prob = model.predict_proba(vec)[0]
        confidence = max(prob)
        is_correct = (pred == expected)
        correct += is_correct
        print(f"Example {i}:")
        print(f"Text: {text}")
        print(f"Expected: {'Real' if expected==1 else 'Fake'}")
        print(f"Predicted: {'Real' if pred==1 else 'Fake'} with {confidence:.2f} confidence")
        print(f"Correct: {is_correct}")
        print("-"*50)

    print(f"Real-world test accuracy: {correct}/{len(examples)} = {correct/len(examples):.2f}")

# -----------------------
# 6. Main
# -----------------------
def main():
    print("📥 Loading data...")
    df = load_data()

    print("🔎 Creating features...")
    X, y, vectorizer = create_features(df)

    print("✂️ Splitting train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print("⚡ Training Logistic Regression...")
    model = train_model(X_train, y_train)

    print("📊 Evaluating...")
    evaluate(model, X_test, y_test)

    print("🧪 Testing with real-world examples...")
    test_examples(model, vectorizer)

    print("💾 Saving model and vectorizer...")
    joblib.dump(model, "logreg_model.pkl")
    joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
    print("✅ Model and vectorizer saved!")

if __name__ == "__main__":
    main()
