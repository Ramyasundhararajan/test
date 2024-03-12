from flask import Flask, render_template, request
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import time
import pandas as pd
import csv

app = Flask(__name__)

# Load pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Function to read CSV file
def read_csv(file_path):
    reviews = []
    sentiments = []

    try:
        with open(file_path, 'r', encoding='utf-8') as csv_file:
            csv_reader = csv.reader(csv_file)
            
            for row in csv_reader:
                if len(row) == 2:
                    review_text, sentiment = row
                    reviews.append(review_text)
                    sentiments.append(sentiment)
                else:
                    print(f"Skipping row: {row}. It does not have two values.")
    except FileNotFoundError:
        print(f"File not found at path: {file_path}")
    except Exception as e:
        print(f"Error reading CSV file: {e}")

    return reviews, sentiments

# Load CSV file for training
train_file_path = 'amazon_review - Copy.csv'  # Change this to the path of your CSV file
train_reviews, train_sentiments = read_csv(train_file_path)

# Function to predict sentiment for input text
def predict_sentiment(input_text, model, tokenizer):
    # Tokenize the input text
    inputs = tokenizer(input_text, return_tensors="pt", max_length=64, padding=True, truncation=True)
    inputs.to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    _, predicted_class = torch.max(logits, dim=1)
    sentiment = "Positive" if predicted_class.item() == 1 else "Negative"

    return sentiment

# Route for predicting sentiment from typed text
@app.route('/predict_text', methods=['POST'])
def predict_text():
    if request.method == 'POST':
        input_text = request.form.get('input_text', '')  # should match the name attribute in the HTML form

        if not input_text:
            return "Input text is empty."

        # Predict sentiment for input text
        sentiment = predict_sentiment(input_text, model, tokenizer)

        return render_template('text_result.html', input_text=input_text, sentiment=sentiment)

# Function to predict sentiments from a file
def predict_sentiments_from_file(file_path, model, tokenizer):
    try:
        df = pd.read_csv(file_path)
        reviews = df['reviewText'].tolist()

        all_sentiments = []

        for review in reviews:
            try:
                start_time = time.time()

                # Ensure review is a string before calling predict_sentiment
                if not isinstance(review, str):
                    raise ValueError("Review must be a string.")

                sentiment = predict_sentiment(review, model, tokenizer)
                elapsed_time = time.time() - start_time

                all_sentiments.append((review, sentiment, elapsed_time))
            except Exception as e:
                all_sentiments.append((review, f"Error: {str(e)}", 0.0))

        return all_sentiments

    except Exception as e:
        return [(str(e), "", 0.0)] * len(reviews)

# Prediction route for file input
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_file', methods=['GET','POST'])
def predict_file():
    if request.method == 'POST':
        file_path = request.form.get('file_path', '')  # should match the name attribute in the HTML form
        if not file_path:
            return "File path is empty."

        all_sentiments = predict_sentiments_from_file(file_path, model, tokenizer)
        return render_template('result.html', review_path=file_path, all_sentiments=all_sentiments)


if __name__ == '__main__':
    app.run(debug=True)
