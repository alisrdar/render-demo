from flask import Flask, render_template, request
import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification

app = Flask(__name__)

MODEL_PATH = "best_priority_model"

tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

label_map = {
    0: "Low",
    1: "Medium",
    2: "High"
}

def predict_priority(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        confidence, predicted_class = torch.max(probs, dim=1)

    return label_map[predicted_class.item()], round(confidence.item() * 100, 2)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    complaint_text = ""

    if request.method == "POST":
        complaint_text = request.form["complaint"]
        prediction, confidence = predict_priority(complaint_text)

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        complaint_text=complaint_text
    )

if __name__ == "__main__":
    app.run(debug=True)
