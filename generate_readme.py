"""
Generate a README file for the Fake News Detection project
"""

# Fake News Detection Project - README Generator

import os

def generate_readme():
    """Generate a comprehensive README.md file for the project"""
    
    readme_content = """# Fake News Detection System

## Overview
This project implements a machine learning-based fake news detection system using Support Vector Machines (SVM) and ensemble methods. The system analyzes text content to determine whether a news article is likely to be fake or real.

## Features
- Text preprocessing with NLTK
- TF-IDF vectorization for feature extraction
- Multiple classifier models (SVM, Random Forest, Logistic Regression)
- Ensemble voting for improved accuracy
- Interactive web interface with Gradio
- Comprehensive evaluation metrics

## Project Structure
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
"""
    
    # Write the README file
    with open("README.md", "w") as f:
        f.write(readme_content)
    
    print("README.md generated successfully!")

if __name__ == "__main__":
    generate_readme()
