# **Sentiment Analysis with BERT**

This project demonstrates a sentiment analysis system built using a pre-trained BERT model (`cardiffnlp/twitter-roberta-base-sentiment`). It evaluates textual data to classify sentiments into **Negative**, **Neutral**, or **Positive**, and provides insights through evaluation metrics and a confusion matrix visualization.

---

## **Features**
- Sentiment classification using a state-of-the-art BERT model.
- Outputs key evaluation metrics such as accuracy, precision, recall, and F1-score.
- Visualizes the confusion matrix for detailed performance analysis.
- Easily configurable and extensible for custom datasets.

---

## **Project Structure**
```
├── app.py               # Main application script
├── requirements.txt      # List of dependencies
├── README.md             # Project documentation
└── example_dataset.csv   # Example dataset for testing (optional)
```

---

## **Prerequisites**
- Python 3.8 or later
- `pip` for installing dependencies
- CUDA-enabled GPU (optional, for faster inference)

---

## **Installation**

1. Clone the repository:
    ```bash
    git clone https://github.com/<your_username>/<your_repo_name>.git
    cd Intel-Products-Sentiment-and-Trend-Analysis
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Download the necessary SpaCy model:
    ```bash
    python -m spacy download en_core_web_sm
    ```

---

## **Usage**

### **Input Data Requirements**
Prepare your dataset as a CSV file with the following columns:
- `title`: A short title for the review or feedback.
- `content`: The main body of the text for sentiment analysis.
- `rating`: Numerical labels for true sentiment (e.g., 0 = Negative, 1 = Neutral, 2 = Positive).

Example:
| title          | content                       | rating |
|----------------|-------------------------------|--------|
| Great movie!   | I loved it!                   | 2      |
| Not great      | It was boring and predictable | 0      |

### **Run the Script**

1. Save your dataset as `dataset.csv` or any other name.
2. Run the `app.py` script, providing the path to your dataset:
    ```bash
    python app.py --data_path <path_to_dataset.csv>
    ```

### **Outputs**
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score
- **Confusion Matrix**: A visual representation of model performance

---

## **Example Output**

Sample metrics output:
```
Accuracy: 0.8750
Precision: 0.8902
Recall: 0.8750
F1-score: 0.8826
Confusion Matrix:
[[50  5  3]
 [ 4 30  6]
 [ 2  3 40]]
```

## **Customization**
This project is easily extensible. To adapt for other sentiment analysis tasks:
1. Replace the dataset with your custom data.
2. Fine-tune the model if necessary using the `transformers` library.

---

## **Contributing**
Contributions are welcome! Feel free to open issues or submit pull requests.

---

## **Contact**
For any questions or suggestions, feel free to reach out to:
- **Your Name**: [ananahmedomar@gmail.com]
- GitHub: https://github.com/Anan651

---
