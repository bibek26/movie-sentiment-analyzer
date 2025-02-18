from flask import Flask, render_template, request, jsonify
import joblib
from sentiment_analyzer import SentimentAnalyzer
import time

app = Flask(__name__)

# Initialize and load the model
analyzer = SentimentAnalyzer()

# Load the data and train the model on startup
def initialize_model():
    import pandas as pd
    from sklearn.model_selection import train_test_split
    
    # Load and prepare data
    df = pd.read_csv('IMDB_dataset.csv')
    X_train, X_test, y_train, y_test = train_test_split(
        df['review'],
        df['sentiment'],
        test_size=0.3,
        random_state=42
    )
    
    # Train model
    X_train_vectorized, X_test_vectorized = analyzer.preprocess_text(X_train, X_test)
    analyzer.build_model(X_train_vectorized, y_train)
    
    return "Model initialized successfully!"

@app.route('/')
def home():
    return '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Movie Review Sentiment Analyzer</title>
        <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600&display=swap" rel="stylesheet">
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: 'Poppins', sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                color: #fff;
                padding: 2rem;
            }
            
            .container {
                max-width: 800px;
                margin: 0 auto;
                background: rgba(255, 255, 255, 0.1);
                backdrop-filter: blur(10px);
                border-radius: 20px;
                padding: 2rem;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            }
            
            h1 {
                font-size: 2.5rem;
                font-weight: 600;
                margin-bottom: 1.5rem;
                text-align: center;
            }
            
            textarea {
                width: 100%;
                min-height: 150px;
                padding: 1rem;
                border: none;
                border-radius: 10px;
                background: rgba(255, 255, 255, 0.9);
                font-size: 1rem;
                margin: 1rem 0;
                transition: all 0.3s ease;
            }
            
            textarea:focus {
                outline: none;
                box-shadow: 0 0 0 3px rgba(255, 255, 255, 0.4);
                background: rgba(255, 255, 255, 1);
            }
            
            button {
                width: 100%;
                padding: 1rem;
                background: #D44ADE;
                color: white;
                border: none;
                border-radius: 10px;
                font-size: 1.1rem;
                font-weight: 500;
                cursor: pointer;
                transition: all 0.3s ease;
                position: relative;
                overflow: hidden;
            }
            
            button:hover {
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
            }
            
            button:active {
                transform: translateY(0);
            }
            
            .loader {
                display: none;
                border: 4px solid #f3f3f3;
                border-top: 4px solid #3498db;
                border-radius: 50%;
                width: 30px;
                height: 30px;
                animation: spin 1s linear infinite;
                margin: 1rem auto;
            }
            
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            
            .result {
                margin-top: 2rem;
                padding: 1.5rem;
                border-radius: 10px;
                animation: fadeIn 0.5s ease;
                text-align: center;
            }
            
            .positive {
                background: rgba(72, 187, 120, 0.9);
            }
            
            .negative {
                background: rgba(239, 68, 68, 0.9);
            }
            
            .confidence {
                margin-top: 1rem;
                font-size: 0.9rem;
                opacity: 0.8;
            }
            
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(20px); }
                to { opacity: 1; transform: translateY(0); }
            }
            
            @media (max-width: 768px) {
                body {
                    padding: 1rem;
                }
                
                .container {
                    padding: 1rem;
                }
                
                h1 {
                    font-size: 2rem;
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎬 Movie Review Sentiment Analyzer</h1>
            <p>Enter a movie review below to analyze its sentiment:</p>
            <textarea id="review" placeholder="Type your review here..."></textarea>
            <button onclick="analyzeSentiment()">
                <span class="button-text">Analyze Sentiment</span>
                <div class="loader"></div>
            </button>
            <div id="result" class="result"></div>
        </div>
        
        <script>
        function analyzeSentiment() {
            const review = document.getElementById('review').value;
            const button = document.querySelector('button');
            const loader = document.querySelector('.loader');
            const buttonText = document.querySelector('.button-text');
            const resultDiv = document.getElementById('result');
            
            if (!review) {
                alert('Please enter a review!');
                return;
            }
            
            // Show loader and disable button
            button.disabled = true;
            buttonText.style.display = 'none';
            loader.style.display = 'block';
            resultDiv.innerHTML = '';
            
            fetch('/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({review: review}),
            })
            .then(response => response.json())
            .then(data => {
                // Hide loader and enable button
                button.disabled = false;
                buttonText.style.display = 'block';
                loader.style.display = 'none';
                
                // Display result
                resultDiv.className = 'result ' + (data.sentiment === 'Positive' ? 'positive' : 'negative');
                resultDiv.innerHTML = `
                    <h3>Analysis Result:</h3>
                    <p>Sentiment: ${data.sentiment}</p>
                    <p class="confidence">Confidence: ${(data.confidence * 100).toFixed(2)}%</p>
                `;
            })
            .catch(error => {
                // Handle error
                button.disabled = false;
                buttonText.style.display = 'block';
                loader.style.display = 'none';
                resultDiv.innerHTML = '<p style="color: #fff">Error analyzing review. Please try again.</p>';
                console.error('Error:', error);
            });
        }
        </script>
    </body>
    </html>
    '''

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    review = data['review']
    
    # Initialize model
    initialize_model()
    
    # Get prediction
    sentiment, confidence = analyzer.predict_new_review(review)
    
    return jsonify({
        'sentiment': sentiment,
        'confidence': confidence
    })

if __name__ == '__main__':
    print("Initializing model...")
    initialize_model()
    print("Starting web server...")
    app.run(debug=True)
