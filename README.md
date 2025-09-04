# Fake News Detector 🕵️

Hey there! Welcome to my fake news detection project. I built this using Support Vector Machines to help figure out if a news article is real or fake. Pretty cool, right? It even comes with a nice web interface so you can test it out yourself!

## What's Inside? 📂

```
Fake-News-SVM/
├── train_model.py       # This is where the magic happens - trains our AI
├── app.py               # The web app you can play with
├── requirements.txt     # All the stuff you need to install
├── README.md            # You're reading this right now!
├── svm_model.pkl        # The trained brain of our detector
└── tfidf_vectorizer.pkl # Helps convert text to numbers
```

## What Can This Do? ✨

- Cleans up messy text and turns it into something the AI can understand
- Uses a fancy linear SVM algorithm to make predictions
- Has a user-friendly web interface (no coding required to use it!)
- Remembers what it learned so you don't have to retrain it every time
- Shows you how well it's performing with detailed stats

## Getting Started 🚀

First things first, let's get everything installed:

```bash
pip install -r requirements.txt
```

Oh, and you'll need to download some language data (just do this once):
```python
import nltk
nltk.download('stopwords')
```

## How to Use This Thing 🎯

### Want to train your own model?
```bash
python train_model.py
```
Sit back and watch it learn! It'll show you how well it's doing.

### Ready to try the web app?
```bash
python app.py
```
Then open your browser and go to `http://127.0.0.1:7860` - it's like magic!

## The Nerdy Details 🤓

If you're curious about how this actually works under the hood:

- **Algorithm**: Support Vector Machine with Linear Kernel (sounds fancy, works great!)
- **Features**: TF-IDF vectors with up to 5000 features (basically turns words into math)
- **Text Cleaning**: Converts to lowercase, removes weird characters, filters out common words
- **Data Split**: 80% for training, 20% for testing (gotta make sure it actually works!)

## About the Data 📰

You'll need to update the file paths in `train_model.py` to point to your own datasets:
- `Fake.csv`: The fake news articles (unfortunately, there are lots of these)
- `True.csv`: The real news articles (the good stuff!)

## A Quick Note 💭

This is mainly for learning and experimenting. While it's pretty good at what it does, always double-check important news from multiple trusted sources. We're fighting misinformation, but let's be smart about it!
