# Fake News Detection System

## Overview
This project implements a machine learning-based fake news detection system using Support Vector Machines (SVM) and ensemble methods. The system analyzes text content to determine whether a news article is likely to be fake or real.

## One-Click Training and Usage
For the simplest experience, use the one-click launcher:

```bash
python one_click_launcher.py
```

This will:
1. Clean up any previous trained models
2. Train a new model using the streamlined approach
3. Launch the web interface for immediate testing

## Features
- Text preprocessing with NLTK
- TF-IDF vectorization for feature extraction
- Multiple classifier models (SVM, Random Forest, Logistic Regression)
- Ensemble voting for improved accuracy
- Interactive web interface with Gradio
- Comprehensive evaluation metrics

## Project Structure
- `one_click_launcher.py`: **NEW** - Train and launch with one command
- `cleanup.py`: **NEW** - Remove all generated files before training
- `train_streamlined.py`: **NEW** - Fast and efficient training script
- `train_model.py`: Original training script
- `improved_train_model.py`: Enhanced training with multiple models
- `train_ensemble_model.py`: Ensemble model training
- `train_real_news_detector.py`: Specialized model for real news detection
- `test_model.py`: Script to test model on sample examples
- `verify_model.py`: Detailed verification with metrics
- `app.py`: Gradio web interface for interactive usage
- `svm_model.pkl`: Trained model
- `tfidf_vectorizer.pkl`: Fitted TF-IDF vectorizer

## Installation

### Prerequisites
- Python 3.8+
- Required packages:

```bash
pip install -r requirements.txt
```

### Setup
1. Clone the repository
2. Install the required dependencies
3. Run the training script:
   ```bash
   python train_real_news_detector.py
   ```
4. Launch the web interface:
   ```bash
   python app.py
   ```

## Usage
1. Open the web interface in your browser (typically at http://127.0.0.1:7860)
2. Paste the news text you want to analyze
3. Click "Analyze" to get the prediction

## Model Performance
The model achieves approximately 80% accuracy on real-world examples with:
- Good precision for fake news detection
- High recall for real news detection
- Balanced performance across different news types

## Dataset
The model was trained on:
- True.csv: Collection of real news articles
- Fake.csv: Collection of fake news articles
- Custom synthetic examples for improved generalization

## Contributors
- Kartik

## License
This project is licensed under the MIT License - see the LICENSE file for details.
