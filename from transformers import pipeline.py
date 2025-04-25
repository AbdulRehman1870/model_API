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
    
    # Collect results above a threshold (e.g., 0.5)
    detected = []
    for label, score in zip(labels, scores):
        if score >= 0.5:
            detected.append((label, float(score)))
    
    return detected, scores

if __name__ == "__main__":  # ✅ Fixed line
    text = input("Enter text to check for vulgarity: ")
    detected, scores = classify_text(text)
    
    if detected:
        print("\nDetected Vulgar/Offensive Content:")
        for label, score in detected:
            print(f"- {label.title()} (Confidence: {score:.2f})")
    else:
        print("\nNo vulgar/offensive content detected.")

    print("\nRaw Scores for Each Category:")
    for label, score in zip(labels, scores):
        print(f"{label.title()}: {score:.2f}")

    # Model accuracy reference
    print("\n[INFO] Model AUC (accuracy) on Jigsaw Toxic Comments benchmark: 0.98636")
