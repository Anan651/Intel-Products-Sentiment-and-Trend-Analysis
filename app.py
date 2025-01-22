import pandas as pd
import seaborn as sns
import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import argparse

# Check if CUDA is available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load pre-trained sentiment analysis model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment')
model = AutoModelForSequenceClassification.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment')
model.to(device)

# Function to evaluate sentiment using a pre-trained model
def evaluate_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    outputs = model(**inputs)
    logits = outputs.logits
    probabilities = softmax(logits.detach().cpu().numpy()[0])
    prediction = probabilities.argmax()
    return prediction, probabilities

def main(data_path):
    # Load data
    df = pd.read_csv(data_path)

    # Initialize lists to store true labels and predictions
    true_labels = df['rating'].tolist()  # Assume 'rating' column holds true labels
    predictions = []

    for idx, row in df.iterrows():
        text = f"{row['title']} {row['content']}"
        sentiment_prediction, probabilities = evaluate_sentiment(text)
        predictions.append(sentiment_prediction)

    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='weighted')
    conf_matrix = confusion_matrix(true_labels, predictions)

    # Print metrics
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1-score: {f1:.4f}')
    print('Confusion Matrix:')
    print(conf_matrix)

    # Plot confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Neutral', 'Positive'],
                yticklabels=['Negative', 'Neutral', 'Positive'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='Path to the CSV data file')
    args = parser.parse_args()
    main(args.data_path)
