from flask import Flask, request, jsonify

# create a Flask app instance
app = Flask(__name__)
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# Load the model and tokenizer
model_name = "unitary/toxic-bert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Define labels for the output
labels = [
    "toxic", "severe_toxic", "obscene", 
    "threat", "insult", "identity_hate"
]

def classify_text(text):
    # Tokenize the text
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        scores = F.sigmoid(outputs.logits)[0].cpu().numpy()
    
    # Return all label-score pairs
    results = []
    for label, score in zip(labels, scores):
        results.append({"label": label, "score": float(score)})
    return results

@app.route('/classify', methods=['POST'])
def classify():
    # Parse the incoming JSON payload
    data = request.get_json()
    text = data.get('text', '')

    # Call your model function
    results = classify_text(text)

    # Return JSON array of {label, score}
    return jsonify(results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
