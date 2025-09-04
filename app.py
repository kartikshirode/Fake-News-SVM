import joblib
import gradio as gr
import re
from nltk.corpus import stopwords

def load_models():
    """Load the saved model and vectorizer"""
    try:
        model = joblib.load("svm_model.pkl")
        vectorizer = joblib.load("tfidf_vectorizer.pkl")
        return model, vectorizer
    except FileNotFoundError:
        raise Exception("Model files not found. Please train the model first by running train_model.py")

def preprocess_text(text):
    """Preprocess input text similar to training data"""
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and keep only letters and spaces
    text = re.sub(r'[^a-z\s]', '', text)
    
    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    text = " ".join([word for word in text.split() if word not in stop_words])
    
    return text

def predict_news(text):
    """Let's see if this news is real or fake!"""
    if not text.strip():
        return "Hey, you need to give me some text to analyze first!"
    
    try:
        # Load models
        model, vectorizer = load_models()
        
        # Clean up the text (just like we did during training)
        processed_text = preprocess_text(text)
        
        # Convert to TF-IDF vector (turn words into numbers the AI understands)
        text_vector = vectorizer.transform([processed_text])
        
        # Make prediction (this is where the magic happens!)
        prediction = model.predict(text_vector)[0]
        confidence = model.decision_function(text_vector)[0]
        
        # Give a friendly response
        if prediction == 1:
            result = "This looks like REAL news!"
            confidence_text = f"I'm {abs(confidence):.2f} confident about this"
        else:
            result = "This smells like FAKE news!"
            confidence_text = f"I'm {abs(confidence):.2f} confident about this"
        
        return f"{result}\n{confidence_text}"
        
    except Exception as e:
        return f"Oops! Something went wrong: {str(e)}"

# Create our web interface
def create_interface():
    """Build the web app interface - this is the fun part!"""
    interface = gr.Interface(
        fn=predict_news,
        inputs=gr.Textbox(
            lines=5, 
            placeholder="Go ahead, paste any news article or headline here and I'll tell you what I think!",
            label="What news do you want me to check?"
        ),
        outputs=gr.Textbox(label="My Analysis"),
        title="Fake News Detective (Powered by AI)",
        description="""
        Hey there! I'm your friendly neighborhood fake news detector. I've been trained to spot suspicious news articles using some pretty cool machine learning magic.
        
        **How to use me:**
        1. Just paste any news headline or article in the box above
        2. Hit that Submit button and watch me work!
        3. I'll tell you if I think it's real or fake, plus how confident I am
        
        **Quick heads up:** I'm pretty good at this, but I'm not perfect! Always double-check important news from trusted sources. Think of me as your first line of defense against misinformation! 🛡️
        """,
        examples=[
            ["Breaking: Scientists discover new planet in our solar system"],
            ["Local man wins lottery for the third time this year"],
            ["New study shows benefits of regular exercise on mental health"]
        ],
        theme=gr.themes.Soft()
    )
    
    return interface

def main():
    """Let's get this party started!"""
    try:
        # Check if our AI is ready to go
        load_models()
        print("Great! My AI brain is loaded and ready to detect fake news!")
        
        # Fire up the web interface
        interface = create_interface()
        interface.launch(
            share=False,  # Set to True if you want to share this with the world!
            server_name="127.0.0.1",
            server_port=7860
        )
        
    except Exception as e:
        print(f"Whoops! Ran into a problem: {e}")
        print("Looks like you need to train the model first. Run 'python train_model.py' and then come back!")

if __name__ == "__main__":
    main()