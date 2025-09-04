import joblib
import gradio as gr
import re
from nltk.corpus import stopwords
import pandas as pd
import numpy as np

def load_models():
    """Load the saved model and vectorizer"""
    try:
        model = joblib.load("svm_model.pkl")
        vectorizer = joblib.load("tfidf_vectorizer.pkl")
        return model, vectorizer
    except FileNotFoundError:
        raise Exception("Model files not found. Please train the model first by running train_model.py")

def clean_text_conservative(text):
    """Matches the exact preprocessing from train_model.py"""
    if pd.isna(text):
        return ""
    
    text = str(text).lower()
    
    # Remove URLs but keep the context
    text = re.sub(r'http\S+|www\S+|https\S+', ' [URL] ', text, flags=re.MULTILINE)
    
    # Replace email with placeholder
    text = re.sub(r'\S+@\S+', ' [EMAIL] ', text)
    
    # Keep some numbers but remove excessive ones
    text = re.sub(r'\b\d{4,}\b', ' [NUMBER] ', text)  # Replace long numbers
    
    # Keep basic punctuation
    text = re.sub(r'[^a-zA-Z0-9\s\.\!\?\,\;\:]', ' ', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # More conservative stopword removal
    stop_words = set(stopwords.words("english"))
    # Keep important words that might indicate bias or opinion
    keep_words = {
        'not', 'no', 'never', 'nothing', 'nobody', 'neither', 'nowhere', 'none',
        'but', 'however', 'although', 'though', 'yet', 'still', 'nevertheless',
        'very', 'really', 'quite', 'extremely', 'highly', 'completely',
        'must', 'should', 'would', 'could', 'might', 'may',
        'always', 'never', 'often', 'sometimes', 'usually'
    }
    stop_words = stop_words - keep_words
    
    words = text.split()
    text = ' '.join([word for word in words if word not in stop_words and len(word) > 1])
    
    return text

def predict_news(text):
    """Let's see if this news is real or fake - now with improved accuracy!"""
    if not text.strip():
        return "🤔 Hey, you need to give me some text to analyze first!"
    
    # Check if text is too short
    if len(text.strip()) < 20:
        return "⚠️ This text seems too short to analyze properly. Try giving me at least a few sentences!"
    
    try:
        # Load models
        model, vectorizer = load_models()
        
        # Clean up the text (using our improved preprocessing)
        processed_text = clean_text_conservative(text)
        
        # Check if anything remains after preprocessing
        if not processed_text.strip():
            return "🤷 After cleaning the text, there's nothing meaningful left to analyze. Try different text!"
        
        # Convert to TF-IDF vector (turn words into numbers the AI understands)
        text_vector = vectorizer.transform([processed_text])
        
        # Make prediction (this is where the magic happens!)
        prediction = model.predict(text_vector)[0]
        
        # Different models have different ways to get confidence scores
        if hasattr(model, 'decision_function'):
            # For SVM
            confidence = abs(model.decision_function(text_vector)[0])
        elif hasattr(model, 'predict_proba'):
            # For Random Forest, Logistic Regression, etc.
            proba = model.predict_proba(text_vector)[0]
            # Use the probability of the predicted class
            confidence = proba[int(prediction)]
        else:
            confidence = 0.5  # Default if no confidence method available
        
        # Calculate confidence as a percentage
        confidence_score = confidence
        
        # Give a more nuanced response
        if prediction == 1:
            if confidence_score > 0.7:
                result = "✅ This looks like REAL news!"
                confidence_text = f"I'm quite confident about this (score: {confidence_score:.2f})"
            elif confidence_score > 0.5:
                result = "✅ This seems like real news"
                confidence_text = f"But I'm not super confident (score: {confidence_score:.2f})"
            else:
                result = "⚠️ This might be real news"
                confidence_text = f"But my confidence is low (score: {confidence_score:.2f})"
        else:
            if confidence_score > 0.7:
                result = "❌ This smells like FAKE news!"
                confidence_text = f"I'm quite confident about this (score: {confidence_score:.2f})"
            elif confidence_score > 0.5:
                result = "❌ This might be fake news"
                confidence_text = f"But I'm not super confident (score: {confidence_score:.2f})"
            else:
                result = "⚠️ This could be fake news"
                confidence_text = f"But my confidence is low (score: {confidence_score:.2f})"
        
        # Add a helpful reminder
        disclaimer = "\n\n💡 Remember: Always verify important news from multiple trusted sources!"
        
        return f"{result}\n{confidence_text}{disclaimer}"
        
    except Exception as e:
        return f"😅 Oops! Something went wrong: {str(e)}\nTry retraining the model if this keeps happening!"

# Create our web interface
def create_interface():
    """Build the web app interface - this is the fun part!"""
    interface = gr.Interface(
        fn=predict_news,
        inputs=gr.Textbox(
            lines=5, 
            placeholder="Paste any news article or headline here (at least a few sentences work best!)",
            label="What news do you want me to check?"
        ),
        outputs=gr.Textbox(label="My Analysis"),
        title="🕵️ Improved Fake News Detective (Now More Accurate!)",
        description="""
        Hey there! I'm your improved fake news detector! I've been retrained with better techniques to work much better on real-world news articles.
        
        **What's new in this version:**
        - Better text preprocessing that removes dataset-specific patterns
        - Improved feature extraction with regularization
        - More robust training with cross-validation
        - Should work MUCH better on real news articles!
        
        **How to use me:**
        1. Paste any news headline or article (at least a few sentences)
        2. Hit Submit and I'll analyze it for you
        3. I'll give you my verdict with a confidence score
        
        **New features:**
        - More nuanced confidence reporting
        - Better handling of edge cases
        - Improved accuracy on real-world data
        
        **Remember:** I'm much better now, but still not perfect! Always verify important news from multiple trusted sources. 🛡️
        """,
        examples=[
            ["Scientists at MIT have developed a new breakthrough in quantum computing that could revolutionize data processing within the next decade."],
            ["SHOCKING: Local woman discovers doctors hate her for this one simple trick that celebrities use to stay young forever!"],
            ["The Federal Reserve announced today a 0.25% interest rate increase following their monthly policy meeting, citing ongoing inflation concerns."]
        ],
        theme=gr.themes.Soft()
    )
    
    return interface

def main():
    """Let's get this party started!"""
    try:
        # Check if our AI is ready to go
        load_models()
        print("🎉 Great! My improved AI brain is loaded and ready to detect fake news!")
        print("🚀 This version should work MUCH better on real-world data!")
        
        # Fire up the web interface
        interface = create_interface()
        interface.launch(
            share=False,  # Set to True if you want to share this with the world!
            server_name="127.0.0.1",
            server_port=7860
        )
        
    except Exception as e:
        print(f"😅 Whoops! Ran into a problem: {e}")
        print("💡 Looks like you need to train the improved model first. Run 'python train_model.py' and then come back!")

if __name__ == "__main__":
    main()
