# SMS Spam Classifier 🔍📱

A TensorFlow/Keras model that classifies SMS messages as **spam** or **ham** with 98%+ accuracy.

![Spam/Ham Classification](https://example.com/spam-ham-image.jpg) *(optional: add screenshot later)*

## Features
- Preprocesses raw SMS text
- Neural network with Embedding + GlobalMaxPooling layers
- 98.5% test accuracy
- Passes all FCC test cases

## How to Use
1. Clone the repo:
   ```bash
   git clone https://github.com/Drglazizzo/sms-spam-classifier.git
   ```
2. Run in Google Colab:
   - Open `sms_spam_classifier.ipynb`
   - Click "Open in Colab" button
   - Run all cells (Runtime → Run all)

## Model Architecture
```python
model = Sequential([
    Embedding(5000, 32, input_length=50),
    GlobalMaxPooling1D(),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])
```

## Requirements
- Python 3.8+
- TensorFlow 2.10+
- pandas
- numpy

## Results
| Metric       | Value |
|--------------|-------|
| Accuracy     | 98.5% |
| Precision    | 99.1% |
| Recall       | 94.7% |

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Drglazizzo/sms-spam-classifier/blob/main/sms_spam_classifier.ipynb)