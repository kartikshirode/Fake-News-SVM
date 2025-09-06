"""
Model Diagnosis Script
Checks for class bias (predicting everything as one class)
"""
import joblib
import re
import pandas as pd
from nltk.corpus import stopwords
import os

# Simple conservative cleaner (match training)
KEEP_WORDS = {
    'not','no','never','nothing','nobody','neither','nowhere','none',
    'but','however','although','though','yet','still','nevertheless',
    'very','really','quite','extremely','highly','completely',
    'must','should','would','could','might','may',
    'always','never','often','sometimes','usually'
}

STOP_WORDS = set(stopwords.words('english')) - KEEP_WORDS

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', ' [URL] ', text)
    text = re.sub(r'\S+@\S+', ' [EMAIL] ', text)
    text = re.sub(r'\b\d{4,}\b', ' [NUMBER] ', text)
    text = re.sub(r'[^a-zA-Z0-9\s\.\!\?\,\;\:]', ' ', text)
    text = ' '.join(text.split())
    words = [w for w in text.split() if w not in STOP_WORDS and len(w)>1]
    return ' '.join(words)

# Load model
if not os.path.exists('svm_model.pkl'):
    print('Model file svm_model.pkl not found. Train the model first.')
    raise SystemExit

model = joblib.load('svm_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

print('Loaded model. Diagnosing...')

examples = [
    ('REAL','Scientists at MIT have developed a new breakthrough in quantum computing that could revolutionize data processing within the next decade.'),
    ('FAKE','SHOCKING: Local woman discovers doctors hate her for this one simple trick that celebrities use to stay young forever!'),
    ('REAL','The Federal Reserve announced today a 0.25% interest rate increase following their monthly policy meeting, citing ongoing inflation concerns.'),
    ('FAKE','Breaking: Government confirms secret alien base under the White House operating since the 1960s.'),
    ('REAL','A peer-reviewed study published in Nature shows promising results for a new Alzheimer treatment after phase two trials.'),
    ('FAKE','Scientists confirm that drinking vinegar every morning rewires your DNA and adds 30 years to your life.'),
]

pred_labels = []

for expected, text in examples:
    ct = clean_text(text)
    X = vectorizer.transform([ct])
    pred = model.predict(X)[0]
    label = 'REAL' if pred==1 else 'FAKE'
    pred_labels.append(label)
    if hasattr(model,'decision_function'):
        dist = model.decision_function(X)[0]
        conf = abs(dist)
    elif hasattr(model,'predict_proba'):
        probs = model.predict_proba(X)[0]
        conf = max(probs)
    else:
        conf = 0.5
    print(f'Expected={expected} Pred={label} Conf={conf:.3f}')

# Summary
real_preds = sum(1 for l in pred_labels if l=='REAL')
fake_preds = sum(1 for l in pred_labels if l=='FAKE')
print('\nPrediction distribution:')
print(f'  REAL: {real_preds}')
print(f'  FAKE: {fake_preds}')
if real_preds==0 or fake_preds==0:
    print('\nWARNING: Model is predicting only one class. Likely overfitting or label mismatch.')
    print('Next steps:')
    print(' 1. Verify training labels (0=fake, 1=real) consistent across training and inference.')
    print(' 2. Ensure balancing code did not drop one class accidentally after cleaning.')
    print(' 3. Re-train with simpler preprocessing (no aggressive stopword removal).')
